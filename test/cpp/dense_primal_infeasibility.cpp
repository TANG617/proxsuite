#include <catch2/catch_test_macros.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

namespace {

namespace px = proxsuite::proxqp;

// Check the certificate directly using identity scaling. In both
// representations the constraints are x = b and -2 <= x <= 0. The proposed dual
// direction is y = 1, z = -1, so A^T y + C^T z = 0 and the correct support is b
// + 2.
bool
acceptsInfeasibilityCertificate(bool box, double b)
{
  px::dense::QP<double> qp(1, 1, box ? 0 : 1, box);
  qp.settings.eps_primal_inf = 1e-12;
  qp.work.b_scaled = Eigen::VectorXd::Constant(1, b);

  if (box) {
    qp.work.l_box_scaled = Eigen::VectorXd::Constant(1, -2.0);
    qp.work.u_box_scaled = Eigen::VectorXd::Zero(1);
  } else {
    qp.work.l_scaled = Eigen::VectorXd::Constant(1, -2.0);
    qp.work.u_scaled = Eigen::VectorXd::Zero(1);
  }

  Eigen::VectorXd dy = Eigen::VectorXd::Ones(1);
  Eigen::VectorXd dz = -dy;
  Eigen::VectorXd ATdy = dy;
  Eigen::VectorXd CTdz = dz;
  auto view = [](Eigen::VectorXd& v) {
    return px::VectorViewMut<double>{ px::from_eigen, v };
  };
  return px::dense::global_primal_residual_infeasibility(view(ATdy),
                                                         view(CTdz),
                                                         view(dy),
                                                         view(dz),
                                                         qp.work,
                                                         qp.model,
                                                         qp.settings,
                                                         box,
                                                         qp.ruiz);
}

} // namespace

TEST_CASE("dense primal certificate: feasible linear bounds")
{
  REQUIRE(!acceptsInfeasibilityCertificate(false, -1.0));
}

TEST_CASE("dense primal certificate: infeasible linear bounds")
{
  REQUIRE(acceptsInfeasibilityCertificate(false, -3.0));
}

TEST_CASE("dense primal certificate: feasible variable bounds")
{
  REQUIRE(!acceptsInfeasibilityCertificate(true, -1.0));
}

TEST_CASE("dense primal certificate: infeasible variable bounds")
{
  REQUIRE(acceptsInfeasibilityCertificate(true, -3.0));
}
