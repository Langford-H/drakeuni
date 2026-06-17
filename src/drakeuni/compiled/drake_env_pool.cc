#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "drake/geometry/scene_graph.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/geometry_set.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/joint_actuator.h"
#include "drake/multibody/tree/model_instance.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace py = pybind11;

namespace {

constexpr int kRootQposDim = 7;
constexpr int kRootQvelDim = 6;

enum SensorKind {
  kGyro = 0,
  kLocalLinvel = 1,
  kGlobalLinvel = 2,
  kGlobalAngvel = 3,
  kBasePosition = 4,
  kUpVector = 5,
  kBaseQuat = 6,
  kDofPos = 7,
  kDofVel = 8,
  kTrackedFramePosition = 9,
  kContactForce = 10,
};

using drake::geometry::CollisionFilterDeclaration;
using drake::geometry::GeometrySet;
using drake::geometry::SceneGraph;
using drake::math::RigidTransform;
using drake::multibody::BodyIndex;
using drake::multibody::ContactResults;
using drake::multibody::ContactModel;
using drake::multibody::DiscreteContactApproximation;
using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::JointActuatorIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::PdControllerGains;
using drake::multibody::RigidBody;
using drake::multibody::SpatialForce;
using drake::systems::Context;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;

py::array_t<double> MakeArray(const std::vector<py::ssize_t>& shape) {
  return py::array_t<double>(shape);
}

Eigen::VectorXd MujocoQposToDrake(const double* qpos, int nq) {
  Eigen::VectorXd out(nq);
  out.segment(0, 4) = Eigen::Map<const Eigen::Vector4d>(qpos + 3);
  out.segment(4, 3) = Eigen::Map<const Eigen::Vector3d>(qpos);
  if (nq > kRootQposDim) {
    out.segment(kRootQposDim, nq - kRootQposDim) =
        Eigen::Map<const Eigen::VectorXd>(qpos + kRootQposDim, nq - kRootQposDim);
  }
  return out;
}

Eigen::VectorXd MujocoQvelToDrake(const double* qvel, int nv) {
  Eigen::VectorXd out(nv);
  out.segment(0, 3) = Eigen::Map<const Eigen::Vector3d>(qvel + 3);
  out.segment(3, 3) = Eigen::Map<const Eigen::Vector3d>(qvel);
  if (nv > kRootQvelDim) {
    out.segment(kRootQvelDim, nv - kRootQvelDim) =
        Eigen::Map<const Eigen::VectorXd>(qvel + kRootQvelDim, nv - kRootQvelDim);
  }
  return out;
}

void DrakeQposToMujoco(const Eigen::VectorXd& qpos, double* out) {
  Eigen::Map<Eigen::Vector3d> pos(out);
  Eigen::Map<Eigen::Vector4d> quat(out + 3);
  pos = qpos.segment(4, 3);
  quat = qpos.segment(0, 4);
  if (qpos.size() > kRootQposDim) {
    Eigen::Map<Eigen::VectorXd> joints(out + kRootQposDim, qpos.size() - kRootQposDim);
    joints = qpos.segment(kRootQposDim, qpos.size() - kRootQposDim);
  }
}

void DrakeQvelToMujoco(const Eigen::VectorXd& qvel, double* out) {
  Eigen::Map<Eigen::Vector3d> linear(out);
  Eigen::Map<Eigen::Vector3d> angular(out + 3);
  linear = qvel.segment(3, 3);
  angular = qvel.segment(0, 3);
  if (qvel.size() > kRootQvelDim) {
    Eigen::Map<Eigen::VectorXd> joints(out + kRootQvelDim, qvel.size() - kRootQvelDim);
    joints = qvel.segment(kRootQvelDim, qvel.size() - kRootQvelDim);
  }
}

void RequireShape(const py::buffer_info& info, const std::vector<py::ssize_t>& shape,
                  const std::string& name) {
  if (info.ndim != static_cast<py::ssize_t>(shape.size())) {
    throw std::invalid_argument(name + " has wrong rank");
  }
  for (int i = 0; i < info.ndim; ++i) {
    if (info.shape[i] != shape[i]) {
      throw std::invalid_argument(name + " has wrong shape");
    }
  }
}

struct ThreadWorkspace {
  Context<double>* plant_context{};
  std::unique_ptr<Simulator<double>> simulator;
};

class DrakeEnvPool {
 public:
  DrakeEnvPool(const std::string& model_file, int nbatch, double sim_dt,
                     py::array_t<double, py::array::c_style | py::array::forcecast> ctrl_limits,
                     py::array_t<double, py::array::c_style | py::array::forcecast> torque_limits,
                     int base_body_index, int push_body_index,
                     const std::vector<int>& tracked_body_indices,
                     py::array_t<double, py::array::c_style | py::array::forcecast> tracked_point_offsets,
                     py::array_t<int, py::array::c_style | py::array::forcecast> sensor_type,
                     py::array_t<int, py::array::c_style | py::array::forcecast> sensor_index,
                     py::array_t<int, py::array::c_style | py::array::forcecast> sensor_adr,
                     py::array_t<int, py::array::c_style | py::array::forcecast> sensor_dim,
                     int nsensordata,
                     double kp, double kd, int nthread)
      : nbatch_(nbatch),
        sim_dt_(sim_dt),
        ctrl_limits_(std::move(ctrl_limits)),
        torque_limits_(std::move(torque_limits)),
        tracked_body_indices_(tracked_body_indices),
        tracked_point_offsets_(std::move(tracked_point_offsets)),
        sensor_type_(std::move(sensor_type)),
        sensor_index_(std::move(sensor_index)),
        sensor_adr_(std::move(sensor_adr)),
        sensor_dim_(std::move(sensor_dim)),
        nsensordata_(nsensordata),
        kp_(kp),
        kd_(kd) {
    if (nbatch_ < 1) {
      throw std::invalid_argument("nbatch must be >= 1");
    }
    nthread_ = std::max(1, std::min(nbatch_, std::max(1, nthread)));
    auto ctrl_info = ctrl_limits_.request();
    if (ctrl_info.ndim != 2 || ctrl_info.shape[1] != 2) {
      throw std::invalid_argument("ctrl_limits must have shape (nu, 2)");
    }
    nu_ = static_cast<int>(ctrl_info.shape[0]);
    RequireShape(torque_limits_.request(), {nu_}, "torque_limits");
    RequireShape(tracked_point_offsets_.request(),
                 {static_cast<py::ssize_t>(tracked_body_indices_.size()), 3},
                 "tracked_point_offsets");
    const auto sensor_type_info = sensor_type_.request();
    if (sensor_type_info.ndim != 1) {
      throw std::invalid_argument("sensor_type must be one-dimensional");
    }
    const py::ssize_t sensor_count = sensor_type_info.shape[0];
    sensor_count_ = static_cast<int>(sensor_count);
    RequireShape(sensor_index_.request(), {sensor_count}, "sensor_index");
    RequireShape(sensor_adr_.request(), {sensor_count}, "sensor_adr");
    RequireShape(sensor_dim_.request(), {sensor_count}, "sensor_dim");
    if (nsensordata_ < 0) {
      throw std::invalid_argument("nsensordata must be non-negative");
    }

    DiagramBuilder<double> builder;
    auto [plant_ref, scene_graph_ref] =
        drake::multibody::AddMultibodyPlantSceneGraph(&builder, sim_dt_);
    plant_ = &plant_ref;
    scene_graph_ = &scene_graph_ref;
    plant_->set_contact_model(ContactModel::kPointContactOnly);
    plant_->set_discrete_contact_approximation(DiscreteContactApproximation::kSap);
    plant_->set_penetration_allowance(1.0e-4);
    const auto model_instances = Parser(plant_).AddModels(model_file);
    if (model_instances.size() != 1) {
      throw std::runtime_error("DrakeEnvPool expected exactly one model instance");
    }
    model_instance_ = model_instances.at(0);

    auto torque = torque_limits_.unchecked<1>();
    for (int i = 0; i < nu_; ++i) {
      auto& actuator = plant_->get_mutable_joint_actuator(JointActuatorIndex(i));
      actuator.set_effort_limit(torque(i));
      actuator.set_controller_gains(PdControllerGains(kp_, kd_));
    }

    // Drake's MJCF parser does not honor MuJoCo contype/conaffinity here, so
    // exclude robot self-collisions before Finalize().
    num_filtered_geometries_ = ExcludeRobotSelfCollisions();
    plant_->Finalize();
    base_body_ = &plant_->get_body(BodyIndex(base_body_index));
    push_body_ = &plant_->get_body(BodyIndex(push_body_index));
    for (int tracked_body_index : tracked_body_indices_) {
      tracked_bodies_.push_back(&plant_->get_body(BodyIndex(tracked_body_index)));
    }
    diagram_ = builder.Build();

    nq_ = plant_->num_positions();
    nv_ = plant_->num_velocities();
    state_dim_ = 1 + nq_ + nv_;
    if (nu_ != plant_->num_actuators()) {
      throw std::runtime_error("ctrl_limits length does not match plant actuators");
    }
    ValidateSensorLayout();
    compact_state_.assign(static_cast<std::size_t>(nbatch_) * state_dim_, 0.0);
    workspaces_.reserve(nthread_);
    for (int i = 0; i < nthread_; ++i) {
      workspaces_.push_back(MakeWorkspace());
    }
  }

  int nbatch() const { return nbatch_; }
  int state_dim() const { return state_dim_; }
  int control_dim() const { return nu_; }
  int nsensordata() const { return nsensordata_; }
  int nthread() const { return nthread_; }
  int workspace_count() const { return static_cast<int>(workspaces_.size()); }
  int num_filtered_geometries() const { return num_filtered_geometries_; }

  py::dict step(py::array_t<double, py::array::c_style | py::array::forcecast> state0,
                int nstep,
                py::array_t<double, py::array::c_style | py::array::forcecast> control,
                py::object push_force) {
    if (nstep < 1) {
      throw std::invalid_argument("nstep must be >= 1");
    }
    auto state_info = state0.request();
    RequireShape(state_info, {nbatch_, state_dim_}, "state0");
    auto control_info = control.request();
    const bool control_is_traj = control_info.ndim == 3;
    if (control_is_traj) {
      RequireShape(control_info, {nbatch_, nstep, nu_}, "control");
    } else {
      RequireShape(control_info, {nbatch_, nu_}, "control");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> push_array;
    bool has_push = !push_force.is_none();
    if (has_push) {
      push_array = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
          push_force);
      RequireShape(push_array.request(), {nbatch_, 3}, "push_force");
    }

    auto state_out = MakeArray({nbatch_, state_dim_});
    auto start = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        for (int env_index = begin; env_index < end; ++env_index) {
          StepOne(workspace, env_index, state0, control, control_is_traj, nstep,
                  has_push ? &push_array : nullptr);
          WriteState(workspace, env_index, state_out);
        }
      };
      RunChunks(worker);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double step_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(elapsed).count();

    py::dict timing;
    timing["step_ms"] = step_ms;
    py::dict output;
    output["state"] = state_out;
    output["timing"] = timing;
    return output;
  }

  py::dict reset(py::array_t<int, py::array::c_style | py::array::forcecast> env_ids,
                 py::array_t<double, py::array::c_style | py::array::forcecast> initial_state) {
    auto ids_info = env_ids.request();
    if (ids_info.ndim != 1) {
      throw std::invalid_argument("env_ids must be one-dimensional");
    }
    const int rows = static_cast<int>(ids_info.shape[0]);
    RequireShape(initial_state.request(), {rows, state_dim_}, "initial_state");
    {
      py::gil_scoped_release release;
      auto ids = env_ids.unchecked<1>();
      auto state = initial_state.unchecked<2>();
      for (int row = 0; row < rows; ++row) {
        const int env_index = ids(row);
        if (env_index < 0 || env_index >= nbatch_) {
          throw std::out_of_range("env_id out of range");
        }
        SaveCompactState(env_index, &state(row, 0));
      }
    }
    return Snapshot();
  }

  py::dict snapshot() { return Snapshot(); }

  py::array_t<double> forward(
      py::array_t<double, py::array::c_style | py::array::forcecast> state0) {
    RequireShape(state0.request(), {nbatch_, state_dim_}, "state0");
    auto sensor_data = MakeArray({nbatch_, nsensordata_});
    {
      py::gil_scoped_release release;
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        auto state = state0.unchecked<2>();
        for (int env_index = begin; env_index < end; ++env_index) {
          LoadState(workspace, &state(env_index, 0));
          WriteSensorRow(workspace, env_index, sensor_data);
        }
      };
      RunChunks(worker);
    }
    return sensor_data;
  }

  py::dict compute_body_state(
      py::array_t<double, py::array::c_style | py::array::forcecast> state0,
      py::array_t<int, py::array::c_style | py::array::forcecast> body_indices) {
    RequireShape(state0.request(), {nbatch_, state_dim_}, "state0");
    const auto body_info = body_indices.request();
    if (body_info.ndim != 1) {
      throw std::invalid_argument("body_indices must be one-dimensional");
    }
    const int body_count = static_cast<int>(body_info.shape[0]);
    auto body_view = body_indices.unchecked<1>();
    std::vector<const RigidBody<double>*> bodies;
    bodies.reserve(body_count);
    for (int i = 0; i < body_count; ++i) {
      const int body_index = body_view(i);
      if (body_index < 0 || body_index >= plant_->num_bodies()) {
        throw std::out_of_range("body index out of range");
      }
      bodies.push_back(&plant_->get_body(BodyIndex(body_index)));
    }

    auto pos = MakeArray({nbatch_, body_count, 3});
    auto quat = MakeArray({nbatch_, body_count, 4});
    auto linvel = MakeArray({nbatch_, body_count, 3});
    auto angvel = MakeArray({nbatch_, body_count, 3});
    {
      py::gil_scoped_release release;
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        auto state = state0.unchecked<2>();
        for (int env_index = begin; env_index < end; ++env_index) {
          LoadState(workspace, &state(env_index, 0));
          WriteBodyStateRow(workspace, env_index, bodies, pos, quat, linvel, angvel);
        }
      };
      RunChunks(worker);
    }
    py::dict output;
    output["pos"] = pos;
    output["quat"] = quat;
    output["linvel"] = linvel;
    output["angvel"] = angvel;
    return output;
  }

 private:
  ThreadWorkspace MakeWorkspace() {
    ThreadWorkspace runtime;
    auto context = diagram_->CreateDefaultContext();
    runtime.plant_context = &plant_->GetMyMutableContextFromRoot(context.get());
    plant_->get_actuation_input_port(model_instance_)
        .FixValue(runtime.plant_context, Eigen::VectorXd::Zero(nu_));
    SetPdTarget(Eigen::VectorXd::Zero(nu_), runtime.plant_context);
    runtime.simulator = std::make_unique<Simulator<double>>(*diagram_, std::move(context));
    runtime.plant_context =
        &plant_->GetMyMutableContextFromRoot(&runtime.simulator->get_mutable_context());
    runtime.simulator->set_target_realtime_rate(0.0);
    runtime.simulator->Initialize();
    return runtime;
  }

  void LoadState(ThreadWorkspace& runtime, const double* state_row) {
    runtime.simulator->get_mutable_context().SetTime(state_row[0]);
    plant_->SetPositions(runtime.plant_context, MujocoQposToDrake(state_row + 1, nq_));
    plant_->SetVelocities(runtime.plant_context, MujocoQvelToDrake(state_row + 1 + nq_, nv_));
    if (nu_ > 0) {
      Eigen::VectorXd target = Eigen::Map<const Eigen::VectorXd>(state_row + 1 + kRootQposDim, nu_);
      SetPdTarget(target, runtime.plant_context);
    }
    runtime.simulator->Initialize();
  }

  void StepOne(ThreadWorkspace& runtime, int env_index,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& state0,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& control,
               bool control_is_traj, int nstep,
               const py::array_t<double, py::array::c_style | py::array::forcecast>* push_force) {
    auto state = state0.unchecked<2>();
    LoadState(runtime, &state(env_index, 0));
    auto ctrl_limits = ctrl_limits_.unchecked<2>();
    for (int substep = 0; substep < nstep; ++substep) {
      Eigen::VectorXd target(nu_);
      if (control_is_traj) {
        auto control_values = control.unchecked<3>();
        for (int j = 0; j < nu_; ++j) {
          target[j] = std::clamp(control_values(env_index, substep, j), ctrl_limits(j, 0),
                                 ctrl_limits(j, 1));
        }
      } else {
        auto control_values = control.unchecked<2>();
        for (int j = 0; j < nu_; ++j) {
          target[j] = std::clamp(control_values(env_index, j), ctrl_limits(j, 0),
                                 ctrl_limits(j, 1));
        }
      }
      SetPdTarget(target, runtime.plant_context);
      if (push_force != nullptr) {
        auto push_values = push_force->unchecked<2>();
        SetExternalPushForce(runtime, push_values(env_index, 0), push_values(env_index, 1),
                             push_values(env_index, 2));
      } else {
        SetExternalPushForce(runtime, 0.0, 0.0, 0.0);
      }
      runtime.simulator->AdvanceTo(runtime.simulator->get_context().get_time() + sim_dt_);
    }
  }

  void SetPdTarget(const Eigen::VectorXd& target_q, Context<double>* plant_context) {
    Eigen::VectorXd desired(2 * nu_);
    desired.head(nu_) = target_q;
    desired.tail(nu_).setZero();
    plant_->get_desired_state_input_port(model_instance_).FixValue(plant_context, desired);
  }

  void SetExternalPushForce(ThreadWorkspace& runtime, double fx, double fy, double fz) {
    std::vector<ExternallyAppliedSpatialForce<double>> forces;
    if (std::abs(fx) > 0.0 || std::abs(fy) > 0.0 || std::abs(fz) > 0.0) {
      ExternallyAppliedSpatialForce<double> applied;
      applied.body_index = push_body_->index();
      applied.p_BoBq_B.setZero();
      applied.F_Bq_W = SpatialForce<double>(Eigen::Vector3d::Zero(), Eigen::Vector3d(fx, fy, fz));
      forces.push_back(applied);
    }
    plant_->get_applied_spatial_force_input_port().FixValue(runtime.plant_context, forces);
  }

  void WriteState(const ThreadWorkspace& runtime, int env_index, py::array_t<double>& state_out) {
    auto state = state_out.mutable_unchecked<2>();
    state(env_index, 0) = runtime.simulator->get_context().get_time();
    Eigen::VectorXd q = plant_->GetPositions(*runtime.plant_context);
    Eigen::VectorXd v = plant_->GetVelocities(*runtime.plant_context);
    DrakeQposToMujoco(q, &state(env_index, 1));
    DrakeQvelToMujoco(v, &state(env_index, 1 + nq_));
    SaveCompactState(env_index, &state(env_index, 0));
  }

  void ValidateSensorLayout() const {
    auto type = sensor_type_.unchecked<1>();
    auto index = sensor_index_.unchecked<1>();
    auto adr = sensor_adr_.unchecked<1>();
    auto dim = sensor_dim_.unchecked<1>();
    for (int i = 0; i < sensor_count_; ++i) {
      if (adr(i) < 0 || dim(i) < 0 || adr(i) + dim(i) > nsensordata_) {
        throw std::invalid_argument("sensor layout exceeds nsensordata");
      }
      switch (type(i)) {
        case kBaseQuat:
          if (dim(i) != 4) {
            throw std::invalid_argument("base quaternion sensor must have dim 4");
          }
          break;
        case kDofPos:
        case kDofVel:
          if (dim(i) != nu_) {
            throw std::invalid_argument("joint state sensor dim must match control_dim");
          }
          break;
        case kTrackedFramePosition:
        case kContactForce:
          if (dim(i) != 3) {
            throw std::invalid_argument("tracked/contact sensor dim must be 3");
          }
          if (index(i) >= static_cast<int>(tracked_bodies_.size())) {
            throw std::invalid_argument("tracked/contact sensor index is out of range");
          }
          break;
        case kGyro:
        case kLocalLinvel:
        case kGlobalLinvel:
        case kGlobalAngvel:
        case kBasePosition:
        case kUpVector:
          if (dim(i) != 3) {
            throw std::invalid_argument("base vector sensor dim must be 3");
          }
          break;
        default:
          throw std::invalid_argument("unknown DrakeUni sensor type");
      }
    }
  }

  void WriteSensorRow(const ThreadWorkspace& runtime, int env_index,
                      py::array_t<double>& sensor_data) const {
    const RigidTransform<double> x_wb =
        plant_->EvalBodyPoseInWorld(*runtime.plant_context, *base_body_);
    const Eigen::Matrix3d r_bw = x_wb.rotation().matrix().transpose();
    const auto velocity_w =
        plant_->EvalBodySpatialVelocityInWorld(*runtime.plant_context, *base_body_);
    const Eigen::Vector3d gyro_b = r_bw * velocity_w.rotational();
    const Eigen::Vector3d linvel_b = r_bw * velocity_w.translational();

    Eigen::VectorXd q = plant_->GetPositions(*runtime.plant_context);
    Eigen::VectorXd v = plant_->GetVelocities(*runtime.plant_context);
    std::vector<Eigen::Vector3d> tracked_positions(
        tracked_bodies_.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> contact_forces(
        tracked_bodies_.size(), Eigen::Vector3d::Zero());
    auto tracked_point_offsets = tracked_point_offsets_.unchecked<2>();
    for (int point = 0; point < static_cast<int>(tracked_bodies_.size()); ++point) {
      const RigidTransform<double> x_wf =
          plant_->EvalBodyPoseInWorld(*runtime.plant_context, *tracked_bodies_[point]);
      const Eigen::Vector3d offset(tracked_point_offsets(point, 0),
                                   tracked_point_offsets(point, 1),
                                   tracked_point_offsets(point, 2));
      tracked_positions[point] = x_wf.translation() + x_wf.rotation().matrix() * offset;
    }

    const auto& contact_results =
        plant_->get_contact_results_output_port().Eval<ContactResults<double>>(
            *runtime.plant_context);
    for (int i = 0; i < contact_results.num_point_pair_contacts(); ++i) {
      const auto& contact = contact_results.point_pair_contact_info(i);
      for (int point = 0; point < static_cast<int>(tracked_bodies_.size()); ++point) {
        const BodyIndex tracked_body = tracked_bodies_[point]->index();
        Eigen::Vector3d force = Eigen::Vector3d::Zero();
        if (contact.bodyA_index() == tracked_body) {
          force = -contact.contact_force();
        } else if (contact.bodyB_index() == tracked_body) {
          force = contact.contact_force();
        } else {
          continue;
        }
        contact_forces[point] += force;
      }
    }

    auto sensor = sensor_data.mutable_unchecked<2>();
    auto type = sensor_type_.unchecked<1>();
    auto index = sensor_index_.unchecked<1>();
    auto adr = sensor_adr_.unchecked<1>();
    auto dim = sensor_dim_.unchecked<1>();
    for (int item = 0; item < sensor_count_; ++item) {
      const int start = adr(item);
      switch (type(item)) {
        case kGyro:
          for (int axis = 0; axis < 3; ++axis) sensor(env_index, start + axis) = gyro_b[axis];
          break;
        case kLocalLinvel:
          for (int axis = 0; axis < 3; ++axis) sensor(env_index, start + axis) = linvel_b[axis];
          break;
        case kGlobalLinvel:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = velocity_w.translational()[axis];
          }
          break;
        case kGlobalAngvel:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = velocity_w.rotational()[axis];
          }
          break;
        case kBasePosition:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = x_wb.translation()[axis];
          }
          break;
        case kUpVector:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = x_wb.rotation().matrix()(axis, 2);
          }
          break;
        case kBaseQuat: {
          const auto quat = x_wb.rotation().ToQuaternion();
          sensor(env_index, start + 0) = quat.w();
          sensor(env_index, start + 1) = quat.x();
          sensor(env_index, start + 2) = quat.y();
          sensor(env_index, start + 3) = quat.z();
          break;
        }
        case kDofPos:
          for (int i = 0; i < dim(item); ++i) {
            sensor(env_index, start + i) = q[kRootQposDim + i];
          }
          break;
        case kDofVel:
          for (int i = 0; i < dim(item); ++i) {
            sensor(env_index, start + i) = v[kRootQvelDim + i];
          }
          break;
        case kTrackedFramePosition:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) = tracked_positions[index(item)][axis];
          }
          break;
        case kContactForce:
          for (int axis = 0; axis < 3; ++axis) {
            sensor(env_index, start + axis) =
                index(item) < 0 ? 0.0 : contact_forces[index(item)][axis];
          }
          break;
      }
    }
  }

  void WriteBodyStateRow(const ThreadWorkspace& runtime, int env_index,
                         const std::vector<const RigidBody<double>*>& bodies,
                         py::array_t<double>& pos_out, py::array_t<double>& quat_out,
                         py::array_t<double>& linvel_out,
                         py::array_t<double>& angvel_out) const {
    auto pos = pos_out.mutable_unchecked<3>();
    auto quat = quat_out.mutable_unchecked<3>();
    auto linvel = linvel_out.mutable_unchecked<3>();
    auto angvel = angvel_out.mutable_unchecked<3>();
    for (int body = 0; body < static_cast<int>(bodies.size()); ++body) {
      const RigidTransform<double> x_wb =
          plant_->EvalBodyPoseInWorld(*runtime.plant_context, *bodies[body]);
      const auto velocity_w =
          plant_->EvalBodySpatialVelocityInWorld(*runtime.plant_context, *bodies[body]);
      for (int axis = 0; axis < 3; ++axis) {
        pos(env_index, body, axis) = x_wb.translation()[axis];
        linvel(env_index, body, axis) = velocity_w.translational()[axis];
        angvel(env_index, body, axis) = velocity_w.rotational()[axis];
      }
      const auto q = x_wb.rotation().ToQuaternion();
      quat(env_index, body, 0) = q.w();
      quat(env_index, body, 1) = q.x();
      quat(env_index, body, 2) = q.y();
      quat(env_index, body, 3) = q.z();
    }
  }

  py::dict Snapshot() {
    auto state_out = MakeArray({nbatch_, state_dim_});
    {
      py::gil_scoped_release release;
      auto worker = [&](int thread_index, int begin, int end) {
        auto& workspace = workspaces_.at(thread_index);
        for (int env_index = begin; env_index < end; ++env_index) {
          LoadState(workspace, CompactStateRow(env_index));
          WriteState(workspace, env_index, state_out);
        }
      };
      RunChunks(worker);
    }
    py::dict output;
    output["state"] = state_out;
    return output;
  }

  void SaveCompactState(int env_index, const double* state_row) {
    const auto row_start =
        compact_state_.begin() + static_cast<std::ptrdiff_t>(env_index) * state_dim_;
    std::copy(state_row, state_row + state_dim_, row_start);
  }

  const double* CompactStateRow(int env_index) const {
    return compact_state_.data() + static_cast<std::size_t>(env_index) * state_dim_;
  }

  int ExcludeRobotSelfCollisions() {
    std::vector<const RigidBody<double>*> bodies;
    for (BodyIndex body_index : plant_->GetBodyIndices(model_instance_)) {
      bodies.push_back(&plant_->get_body(body_index));
    }
    int count = 0;
    for (const RigidBody<double>* body : bodies) {
      count += static_cast<int>(plant_->GetCollisionGeometriesForBody(*body).size());
    }
    if (count > 0) {
      GeometrySet robot_geometries = plant_->CollectRegisteredGeometries(bodies);
      scene_graph_->collision_filter_manager().Apply(
          CollisionFilterDeclaration().ExcludeWithin(std::move(robot_geometries)));
    }
    return count;
  }

  template <typename Worker>
  void RunChunks(Worker worker) {
    const int thread_count = std::min(nthread_, nbatch_);
    if (thread_count <= 1) {
      worker(0, 0, nbatch_);
      return;
    }
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int thread = 0; thread < thread_count; ++thread) {
      const int begin = thread * nbatch_ / thread_count;
      const int end = (thread + 1) * nbatch_ / thread_count;
      threads.emplace_back([&, thread, begin, end]() { worker(thread, begin, end); });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  int nbatch_{};
  double sim_dt_{};
  int nq_{};
  int nv_{};
  int nu_{};
  int state_dim_{};
  py::array_t<double> ctrl_limits_;
  py::array_t<double> torque_limits_;
  std::vector<int> tracked_body_indices_;
  py::array_t<double> tracked_point_offsets_;
  py::array_t<int> sensor_type_;
  py::array_t<int> sensor_index_;
  py::array_t<int> sensor_adr_;
  py::array_t<int> sensor_dim_;
  int sensor_count_{};
  int nsensordata_{};
  double kp_{};
  double kd_{};
  int nthread_{};
  int num_filtered_geometries_{};
  std::unique_ptr<Diagram<double>> diagram_;
  MultibodyPlant<double>* plant_{};
  SceneGraph<double>* scene_graph_{};
  ModelInstanceIndex model_instance_;
  const RigidBody<double>* base_body_{};
  const RigidBody<double>* push_body_{};
  std::vector<const RigidBody<double>*> tracked_bodies_;
  std::vector<double> compact_state_;
  std::vector<ThreadWorkspace> workspaces_;
};

bool BatchAvailable() { return true; }

}  // namespace

PYBIND11_MODULE(_drake_env_pool, m) {
  m.doc() = "Compiled DrakeEnvPool for UniLab DrakeUni batch experiments.";
  m.def("batch_available", &BatchAvailable);
  py::class_<DrakeEnvPool>(m, "DrakeEnvPool")
      .def(py::init<const std::string&, int, double,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>, int, int,
                    const std::vector<int>&,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>,
                    py::array_t<int, py::array::c_style | py::array::forcecast>, int, double,
                    double, int>(),
           py::arg("model_file"), py::arg("nbatch"), py::arg("sim_dt"),
           py::arg("ctrl_limits"), py::arg("torque_limits"), py::arg("base_body_index"),
           py::arg("push_body_index"), py::arg("tracked_body_indices"),
           py::arg("tracked_point_offsets"),
           py::arg("sensor_type"), py::arg("sensor_index"), py::arg("sensor_adr"),
           py::arg("sensor_dim"), py::arg("nsensordata"),
           py::arg("kp"), py::arg("kd"), py::arg("nthread") = 1)
      .def_property_readonly("nbatch", &DrakeEnvPool::nbatch)
      .def_property_readonly("state_dim", &DrakeEnvPool::state_dim)
      .def_property_readonly("control_dim", &DrakeEnvPool::control_dim)
      .def_property_readonly("nsensordata", &DrakeEnvPool::nsensordata)
      .def_property_readonly("nthread", &DrakeEnvPool::nthread)
      .def_property_readonly("workspace_count", &DrakeEnvPool::workspace_count)
      .def_property_readonly("num_filtered_geometries",
                             &DrakeEnvPool::num_filtered_geometries)
      .def("step", &DrakeEnvPool::step, py::arg("state0"), py::arg("nstep"),
           py::arg("control"), py::arg("push_force") = py::none())
      .def("forward", &DrakeEnvPool::forward, py::arg("state0"))
      .def("compute_body_state", &DrakeEnvPool::compute_body_state, py::arg("state0"),
           py::arg("body_indices"))
      .def("reset", &DrakeEnvPool::reset, py::arg("env_ids"), py::arg("initial_state"))
      .def("snapshot", &DrakeEnvPool::snapshot);
}
