//! Simple, tunable, physics-based subsonic fixed-wing aerodynamics

/// Abstract description of a glider facing -Z, with +Y up, with the center of lift at the origin
#[derive(Debug, Copy, Clone)]
pub struct Geometry {
    /// Angle of attack (rotation of the main wing around the X axis) that stabilization torque
    /// converges towards
    pub trim: f32,
    /// Effective wing size, linearly proportional to lift (force per airspeed squared)
    pub wing_area: f32,
    /// Strength of parasitic drag in forward flight (force per forward airspeed squared)
    pub drag_area: f32,
    /// Stabilization torque per forward airspeed squared
    pub stabilizer_strength: mint::Vector3<f32>,
    /// Strength of drag incurred by stabilization around each axis (force per airspeed squared)
    pub stabilizer_drag: mint::Vector3<f32>,
    /// Torque produced per unit control input, per forward airspeed squared
    pub control_strength: mint::Vector3<f32>,
}

impl Geometry {
    /// Compute the forces acting on `self` when the medium is moving at `velocity` in local
    /// coordinates
    pub fn simulate(&self, velocity: mint::Vector3<f32>, controls: mint::Vector3<f32>) -> Force {
        let v = na::Vector3::from(velocity);
        // Velocity of airflow over the stabilizers
        let s_v = na::UnitQuaternion::from_axis_angle(&na::Vector3::x_axis(), self.trim) * v;
        // Squared
        let s_v_2 = s_v.component_mul(&s_v.map(|x| x.abs()));
        let stabilization_torque =
            na::Vector3::from(self.stabilizer_strength).component_mul(&na::Vector3::new(
                s_v_2.y,  // Pitch into vertical velocity
                -s_v_2.x, // Yaw into sideways velocity
                s_v_2.x,  // Roll away from sideways velocity
            ));
        let stabilizer_drag = na::Vector3::new(
            self.stabilizer_drag.y * -s_v_2.x + self.stabilizer_drag.z * -s_v_2.x,
            self.stabilizer_drag.x * -s_v_2.y,
            0.0,
        );
        let lift = na::Vector3::new(0.0, self.wing_area * -(v.y * v.y.abs()), 0.0);
        let drag = na::Vector3::new(0.0, 0.0, self.drag_area * -(v.z * v.z.abs()));

        let control_torque = na::Vector3::from(self.control_strength)
            .component_mul(&na::Vector3::from(controls))
            .component_mul(&na::Vector3::new(-s_v_2.z, s_v_2.z, -s_v_2.z));

        Force {
            linear: (lift + stabilizer_drag + drag).into(),
            angular: (stabilization_torque + control_torque).into(),
        }
    }
}

impl Default for Geometry {
    fn default() -> Self {
        Self {
            trim: 0.0,
            wing_area: 0.0,
            drag_area: 0.0,
            stabilizer_strength: [0.0; 3].into(),
            stabilizer_drag: [0.0; 3].into(),
            control_strength: [0.0; 3].into(),
        }
    }
}

/// Forces affecting a glider
///
/// Produced by [`Geometry::simulate`], but may also be updated prior to integration with external
/// forces like gravity and thrust.
#[derive(Debug, Copy, Clone)]
pub struct Force {
    pub linear: mint::Vector3<f32>,
    /// Torque
    pub angular: mint::Vector3<f32>,
}

impl Default for Force {
    fn default() -> Self {
        Self {
            linear: [0.0; 3].into(),
            angular: [0.0; 3].into(),
        }
    }
}

/// Properties of a glider governing integration of forces
#[derive(Debug, Copy, Clone)]
pub struct Dynamics {
    /// Relative to the center of lift
    pub center_of_mass: mint::Vector3<f32>,
    pub mass: f32,
    pub angular_damping: mint::Vector3<f32>,
    pub angular_inertia: f32,
}

impl Default for Dynamics {
    fn default() -> Self {
        Self {
            center_of_mass: [0.0; 3].into(),
            mass: 0.0,
            angular_damping: [0.0; 3].into(),
            angular_inertia: 0.0,
        }
    }
}

/// Instantaneous state of a glider
#[derive(Debug, Copy, Clone)]
pub struct State {
    /// Velocity in local coordinates
    pub velocity: mint::Vector3<f32>,
    pub orientation: mint::Quaternion<f32>,
    pub angular_velocity: mint::Vector3<f32>,
}

impl State {
    pub fn integrate(&mut self, dt: f32, force: &Force, dynamics: &Dynamics) {
        // Add lift/drag, taking care not to accelerate past zero velocity in the direction of
        // net aerodynamic forces.
        let dv = (na::Vector3::from(force.linear) / dynamics.mass) * dt;
        let (drag_dir, max_drag) =
            na::Unit::try_new_and_get(dv, 1e-3).unwrap_or_else(|| (na::Vector3::z_axis(), 0.0));
        let parallel_vel = na::Vector3::from(self.velocity).dot(drag_dir.as_ref());
        let drag = (-parallel_vel).min(max_drag) * drag_dir.as_ref();
        self.velocity = (na::Vector3::from(self.velocity) + drag).into();

        // Add torque due to weight shift
        let drag_force = (drag / dt) * dynamics.mass; // force.linear adjusted not to overshoot
        let torque = na::Vector3::from(force.angular)
            + (-na::Vector3::from(dynamics.center_of_mass)).cross(&drag_force);

        // Integrate torque and rotation
        let orientation = na::Unit::new_unchecked(na::Quaternion::from(self.orientation));
        let angular_accel = orientation * torque * (1.0 / dynamics.angular_inertia);
        let mut angular_velocity = na::Vector3::from(self.angular_velocity);
        angular_velocity += angular_accel * dt;
        angular_velocity.zip_apply(&na::Vector3::from(dynamics.angular_damping), |x, damp| {
            *x *= (-damp * dt).exp()
        });
        self.angular_velocity = angular_velocity.into();
        let mut orientation = na::UnitQuaternion::new(angular_velocity * dt) * orientation;
        orientation.renormalize_fast();
        self.orientation = orientation.into_inner().into();
    }
}

impl Default for State {
    fn default() -> Self {
        Self {
            velocity: [0.0; 3].into(),
            orientation: mint::Quaternion {
                v: [0.0; 3].into(),
                s: 1.0,
            },
            angular_velocity: [0.0; 3].into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use std::f32;

    fn glider() -> Geometry {
        Geometry {
            trim: f32::consts::FRAC_PI_8,
            wing_area: 1.0,
            drag_area: 0.0,
            stabilizer_strength: [1.0; 3].into(),
            stabilizer_drag: [1.0; 3].into(),
            control_strength: [1.0; 3].into(),
        }
    }

    #[test]
    fn level_flight() {
        let velocity = na::UnitQuaternion::from_axis_angle(&na::Vector3::x_axis(), -glider().trim)
            * -na::Vector3::z();
        let f = glider().simulate(velocity.into(), [0.0; 3].into());
        assert_abs_diff_eq!(na::Vector3::from(f.angular), na::Vector3::zeros());
        assert_abs_diff_eq!(na::Vector3::from(f.linear).xz(), na::Vector2::zeros());
        assert!(f.linear.y > 0.0);
    }

    #[test]
    fn pitch_stability() {
        let f = glider().simulate((-na::Vector3::z()).into(), [0.0; 3].into());
        assert_abs_diff_eq!(na::Vector3::from(f.angular).yz(), na::Vector2::zeros());
        assert!(f.angular.x > 0.0);
        assert_abs_diff_eq!(na::Vector3::from(f.linear).xz(), na::Vector2::zeros());
        assert!(f.linear.y < 0.0);
    }

    #[test]
    fn roll_yaw_stability() {
        let velocity = na::UnitQuaternion::from_axis_angle(&na::Vector3::x_axis(), -glider().trim)
            * na::Vector3::new(1.0, 0.0, -1.0);
        let f = glider().simulate(velocity.into(), [0.0; 3].into());
        assert_abs_diff_eq!(f.angular.x, 0.0);
        assert!(f.angular.y < 0.0);
        assert!(f.angular.z > 0.0);
        assert_abs_diff_eq!(f.linear.z, 0.0);
        dbg!(f.linear);
        assert!(f.linear.x < 0.0);
    }

    #[test]
    fn control_signs() {
        let velocity = na::UnitQuaternion::from_axis_angle(&na::Vector3::x_axis(), -glider().trim)
            * na::Vector3::new(0.0, 0.0, -1.0);
        let f = glider().simulate(velocity.into(), [1.0; 3].into());
        assert!(f.angular.x > 0.0);
        assert!(f.angular.y < 0.0);
        assert!(f.angular.z > 0.0);
    }
}
