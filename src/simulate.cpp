#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>

#include <armadillo> 
#include "matplotlibcpp.h"
#include "armalogp.h"

namespace plt = matplotlibcpp;

using namespace arma;
using namespace std;

float pi;
float g22, g45, g67, g90, g112, g135, g157, g180, g202, g225, g247, g270, g292, g315, g337, g360, radian;

float logpdf(const fcolvec& x, const fmat& cov)
{
}

#if 0
template<typename T, typename U>
double multivariate_normal_chol_logp(const T& x, const U& mu, const arma::mat& R)
{
	static double log_2pi = log(2.0f * arma::math::pi());
	double ldet = log(cholesky_determinant(R));
	return -0.5f * (x.n_elem * log_2pi + ldet + mahalanobis_chol(x, mu, R));
}

typedef struct _armalogp_t
{
	static inline double square(double x)
	{
		return x*x;
	}
	static inline int square(int x)
	{
		return x*x;
	}
	static double cholesky_determinant(const arma::mat& R)
	{
		return arma::prod((R.diag()*R.diag()));
	}
	static double mahalanobis(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma)
	{
		const arma::vec err = x - mu;
		return std::sqrt(arma::as_scalar(err.t() * sigma.i() * err));
	}
	static double mahalanobis(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& sigma)
	{
		const arma::rowvec err = x - mu;
		return std::sqrt(arma::as_scalar(err * sigma.i() * err.t()));
	}
	static double mahalanobis_chol(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& R)
	{
		const arma::rowvec err = x - mu;
		const arma::mat Rinv(inv(trimatl(R)));
		return std::sqrt(arma::as_scalar(err * Rinv * Rinv.t() * err.t()));
	}
	static double multivariate_normal_chol_logp(const arma::mat& x, const arma::vec& mu, const arma::mat& R)
	{
		const arma::rowvec mu_r = mu.t();
		double ans(0.0);
		for( size_t i = 0; i < x.n_rows; ++i )
			ans += multivariate_normal_chol_logp(x.row(i), mu_r, R);

		return ans;
	}
	static double multivariate_normal_sigma(const arma::mat& x, const arma::vec& mu, const arma::mat& sigma)
	{
		arma::mat R;
		bool chol_succeeded = chol(R, sigma);

		if(!chol_succeeded )
			return -std::numeric_limits<double>::infinity();

		const arma::rowvec mu_r = mu.t();
		double ans(0.0);
		for( size_t i = 0; i < x.n_rows; ++i )
			ans += multivariate_normal_chol_logp(x.row(i), mu_r, R);

		return ans;
	}
	static double wishart_logp(const arma::mat& X, const arma::mat& tau, const unsigned int n)
	{
		if( (X.n_cols != X.n_rows) || (tau.n_cols != tau.n_rows) || (X.n_cols != tau.n_rows) || (X.n_cols > n) )
			return -std::numeric_limits<double>::infinity();

		const double lg2 = log(2.0f);
		const int k = X.n_cols;
		const double dx(arma::det(X));
		const double db(arma::det(tau));

		if( (dx <= 0.0f) || (db <= 0.0f) )
			return -std::numeric_limits<double>::infinity();

		const double ldx = log(dx);
		const double ldb = log(db);
		const arma::mat bx(X * tau);
		const double tbx = arma::trace(bx);

		double cum_lgamma(0.0);
		for(size_t i = 0; i < X.n_rows; ++i)
			cum_lgamma += lgamma((n+1)/2.0f);

		return (n - k - 1)/ 2.0f * ldx + (n/2.0f)*ldb - 0.5f*tbx - (n*k/2.0f)*lg2 - cum_lgamma;
	}
} armalogp_t;
#endif

#if 0
double wishart_logp(const arma::mat& X, const arma::mat& tau, const unsigned int n)
{
	if( (X.n_cols != X.n_rows) || (tau.n_cols != tau.n_rows) || (X.n_cols != tau.n_rows) || (X.n_cols > n) )
		return -std::numeric_limits<double>::infinity();

	const double lg2 = log(2.0f);
	const int k = X.n_cols;
	const double dx(arma::det(X));
	const double db(arma::det(tau));

	if( (dx <= 0.0f) || (db <= 0.0f) )
		return -std::numeric_limits<double>::infinity();

	const double ldx = log(dx);
	const double ldb = log(db);
	const arma::mat bx(X * tau);
	const double tbx = arma::trace(bx);

	double cum_lgamma(0.0);
	for(size_t i = 0; i < X.n_rows; ++i)
	{
		cum_lgamma += lgamma((n+1)/2.0f);
	}

	return (n - k - 1)/ 2.0f * ldx + (n/2.0f)*ldb - 0.5f*tbx - (n*k/2.0f)*lg2 - cum_lgamma;
}

double multivariate_normal_chol_logp(const arma::mat& x, const arma::vec& mu, const arma::mat& R)
{
	const arma::rowvec mu_r = mu.t();
	double ans(0.0);
	for( size_t i = 0; i < x.n_rows; ++i )
	{
		ans += multivariate_normal_chol_logp(x.row(i), mu_r, R);
	}

	return ans;
}

double multivariate_normal_sigma(const arma::mat& x, const arma::vec& mu, const arma::mat& sigma)
{
	arma::mat R;
	bool chol_succeeded = chol(R, sigma);

	if(!chol_succeeded )
		return -std::numeric_limits<double>::infinity();

	const arma::rowvec mu_r = mu.t();
	double ans(0.0);
	for( size_t i = 0; i < x.n_rows; ++i )
	{
		ans += multivariate_normal_chol_logp(x.row(i), mu_r, R);
	}

	return ans;
}

double mahalanobis(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma)
{
	const arma::vec err = x - mu;
	return std::sqrt(arma::as_scalar(err.t() * sigma.i() * err));
}

double mahalanobis(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& sigma)
{
	const arma::rowvec err = x - mu;
	return std::sqrt(arma::as_scalar(err.t() * sigma.i() * err));
}
#endif

void init_global()
{
	pi = 4.0f * atan(1.0f);

	g45 = pi / 4.0f;
	g90 = pi / 2.0f;
	g135 = g45 + g90;
	g180 = pi;
	g225 = g180 + g45;
	g270 = g90 + g180;
	g315 = g270 + g45;
	g360 = g180 * 2.0f;

	radian = 180.0f / g180;

	g22 = 22.5f / radian;
	g67 = 67.5f / radian;
	g112 = 112.5f / radian;
	g157 = 157.5f / radian;
	g202 = 202.5f / radian;
	g247 = 247.5f / radian;
	g292 = 292.5f / radian;
	g337 = 337.5f / radian;
}

void get_xy(const fvec& rf, fvec& xy)
{
	float ff = rf(1);

	if( rf(0) < 0.0f || ff < 0.0f )
	{
		xy(0) = 0.0f;
		xy(1) = 0.0f;
	}
	if( ff <= g90 )
	{
		xy(0) = rf(0) * sin(ff);
		xy(1) = rf(0) * cos(ff);
	}
	else if( ff <= g180 )
	{
		ff -= g90;
		xy(0) = rf(0) * cos(ff);
		xy(1) = -rf(0) * sin(ff);
	}
	else if( ff <= g270 )
	{
		ff -= g180;
		xy(0) = -rf(0) * sin(ff);
		xy(1) = -rf(0) * cos(ff);
	}
	else if( ff <= g360 )
	{
		ff -= g270;
		xy(0) = -rf(0) * cos(ff);
		xy(1) = rf(0) * sin(ff);
	}	
	else
	{
		ff -= g360;
		xy(0) = rf(0) * sin(ff);
		xy(1) = rf(0) * cos(ff);
	}
}

void get_rf(const fvec& xy, fvec& rf)
{
	rf(0) = std::hypot(xy(0), xy(1));

	if( rf(0) > 0.001f )
	{
		rf(1) = acos(xy(1) / rf(0));
		if( xy(0) < 0.0f )
			rf(1) = g360 - rf(1);
	}
	else
		rf(1) = 0.0f;
}

bool inside(const float left, const float right, const float angle)
{
	bool ret = false;

	float df = right - left;

	if( df > 0.0f )
		ret = (angle >= left) && (angle <= right);
	else if( df < 0.0f )
	{
		if( fabs(df) >= g180 )
			ret = (angle >= left) || (angle <= right);
		else
			ret = (angle >= right) && (angle <= left);
	}
	else
		ret = (angle == left);

	return ret;
}

bool north_crossed(const float f1, const float f2)
{
	bool rc = false;
	float df = fabs(f1 - f2);

	if( df > g180 )
		rc = true;

	return rc;
}

#if 0
float residual(const float a, const float b)
{
	float y = a - b;
	y = y % (2 * pi);
	if( y > pi )
		y -= 2*pi;

	return y;
}

float residual(const fvec& a, const fvec& b)
{
	float y = a[0] - b[0];
	y = y % (2 * pi);
	if( y > pi )
		y -= 2*pi;

	return y;
}
#endif

enum { filter_GH = 0, filter_KF };

typedef struct
{
	void init(const float g, const float h, const float k, const float dt)
	{
		x_ = zeros<fvec>(3);
		x_pred_ = zeros<fvec>(3);

		dt_ = dt;
		g_ = g;
		h_ = h;
		k_ = k;
	}
	void init(const fvec& x0)
	{
		x_(0) = x0(0);
		if( x0.size() > 1 )
			x_(1) = x0(1);
		if( x0.size() > 2 )
			x_(2) = x0(2);

		x_pred_ = x_;
	}
	void init(const float x, const float dx, const float ddx)
	{
		x_(0) = x;
		x_(1) = dx;
		x_(2) = ddx;

		x_pred_ = x_;
	}
	float update(float dt, const float z, float g, float h, float k)
	{
		float y;

		if( dt < 0.0f )
			dt = dt_;
		
		if( g < 0.0f )
			g = g_;

		if( h < 0.0f )
			h = h_;

		if( k < 0.0f )
			k = k_;

		float dt_sqr = dt * dt;
		x_pred_(2) = x_(2);
		x_pred_(1) = x_(1) + x_(2) * dt;
		x_pred_(0) = x_(0) + x_(1) * dt + 0.5f * x_(2) * dt_sqr;

		y = z - x_pred_(0);

		x_(2) = x_pred_(2) + 2.0f * k * y / dt_sqr;
		x_(1) = x_pred_(1) + h * y / dt;
		x_(0) = x_pred_(0) + g * y;

		return x_(0);
	}
	void ons(const float g, float& h, float& k)
	{
		float g2 = g*g;
		float g3 = g2*g;
		float g4 = g2*g2;
		float g5 = g2*g3;
		float g6 = g4*g2;

		h = ((2.0f*g3 - 4.0f*g2) + sqrt(4.0f*g6 -64.0f*g5 + 64.0f*g4)) / (8.0f*(1.0f-g));
		k = (h*(2.0f-g) - g2) / g;
	}
	void dumping(float theta, float &g, float&h, float &k)
	{
		theta = std::max(0.0f, std::min(1.0f, theta));

		float theta2 = theta * theta;
		float theta3 = theta2 * theta;

		g = 1.0f - theta3;
		h = 1.5f * (1.0f - theta2) * (1.0f - theta);

		float th = (1.0f - theta);
		k = 0.5f * th * th * th;
	}
	void vrf(float &vx, float &vdx, float &vddx)
	{
		float hg4 = 4.0f - 2.0f*g_ - h_;
		float ghk = g_*h_ + g_*k_ - 2.0f*k_;

		vx = (2.0f*h_*(2.0f*g_*g_ + 2.0f*h_ - 3.0f*g_*h_) - 2.0f*g_*k_*hg4) / (2.0f*k_ - g_*(h_+k_)*hg4);
		vdx = (2.0f*h_*h_*h_ - 4.0f*h_*h_*k_ + 4.0f*k_*k_*(2.0f-g_)) / (2.0f*hg4*ghk);
		vddx = 8.0f*h_*k_*k_ / (dt_*dt_*dt_*dt_*hg4*ghk);
	}
	float vrf_pred()
	{
		float gh2 = 2.0f*g_ + h_;

		float ret = (g_*k_*(gh2-4.0f) + h_*(g_*gh2+2.0f*h_)) /
			(2.0f*k_ - (g_*(h_+k_)*(gh2-4.0f)));

		return ret;
	}
	fvec x_;
	fvec x_pred_;
	float dt_;
	float g_;
	float h_;
	float k_;
} GHFilter_t;

typedef struct
{
	void init(const int dimx, const int dimz, const int dimu=0)
	{
		alpha_sq = 1.0f;

		dim_x = std::max(1, dimx);
		dim_z = std::max(1, dimz);
		dim_u = std::max(0, dimu);

		_x.zeros(dim_x);
		_P.eye(dim_x, dim_x);
		_Q.eye(dim_x, dim_x);
		if( dim_u > 0 )
			_B.zeros(dim_u, dim_u);
		_F.eye(dim_x, dim_x);
		_H.zeros(dim_z, dim_x);
		_R.eye(dim_z, dim_z);
		M.zeros(dim_z, dim_z);
		_z.zeros(dim_z);
		K.zeros(dim_x, dim_z);
		_y.zeros(dim_z);
		S.zeros(dim_z, dim_z);
		SI.zeros(dim_z, dim_z);
		I.eye(dim_x, dim_x);

		x_prior = _x;
		P_prior = _P;

		x_post = _x;
		P_post = _P;

		log_likelihood = datum::log_min;
		likelihood = exp(datum::log_min);
		mahalanobis = datum::nan;
	}
	void set_alpha(const float alpha)
	{
		float al = std::max(1.0f, alpha);
		alpha_sq = al * al;
	}
	void predict(const fcolvec& u, const fmat& b, const fmat& f, const fmat& q)
	{
		fmat B, F, Q;

		if( false == b.is_empty() )
			B = b;
		else
			B = _B;

		if( false == f.is_empty() )
			F = f;
		else
			F = _F;

		if( false == q.is_empty() )
			Q = q;
		else
			Q = _Q;

		if( (false == u.is_empty()) && (false == B.is_empty()) )
			_x = F*_x + B*u;
		else
			_x = F*_x;

		_P = alpha_sq*F*_P*F.t() + Q;

		x_prior = _x;
		P_prior = _P;
	}
	void update(const fcolvec& z, const fmat& r, const fmat& h)
	{
		fmat R, H;

		log_likelihood = datum::nan;
		likelihood = datum::nan;
		mahalanobis = datum::nan;

		if( false == r.is_empty() )
			R = r;
		else
			R = _R;

		if( false == h.is_empty() )
			H = h;
		else
			H = _H;

		if( true == z.is_empty() )
		{
			_z.reset();
			x_post = _x;
			P_post = _P;
			_y.zeros();
			return;
		}

		_y = z - H*_x;

		fmat PHT = _P*H.t();
		S = H*PHT + R;
		SI = inv(S);
		K = PHT*SI;

		mahalanobis = calc_mahalanobis();

		_x = _x + K*_y;

		fmat I_KH = I - K*H;
		_P = I_KH*_P*I_KH.t() + K*R*K.t();

		_z = z;
		x_post = _x;
		P_post = _P;
	}
	void predict_steadystate(const fcolvec& u, const fmat& b)
	{
		fmat B;

		if( false == b.is_empty() )
			B = b;
		else
			B = _B;

		if( (false == u.is_empty()) && (false == B.is_empty()) )
			_x = _F*_x + B*u;
		else
			_x = _F*_x;

		x_prior = _x;
		P_prior = _P;
	}
	void update_steadystate(const fcolvec& z)
	{
		log_likelihood = datum::nan;
		likelihood = datum::nan;
		mahalanobis = datum::nan;

		if( true == z.is_empty() )
		{
			_z.reset();
			x_post = _x;
			P_post = _P;
			_y.zeros();
			return;
		}

		_y = z - _H*_x;

		_x = _x + K*_y;

		_z = z;
		x_post = _x;
		P_post = _P;
	}
	void update_correlated(const fcolvec& z, const fmat& r, const fmat& h)
	{
		fmat R, H;

		log_likelihood = datum::nan;
		likelihood = datum::nan;
		mahalanobis = datum::nan;

		if( false == r.is_empty() )
			R = r;
		else
			R = _R;

		if( false == h.is_empty() )
			H = h;
		else
			H = _H;

		if( true == z.is_empty() )
		{
			_z.reset();
			x_post = _x;
			P_post = _P;
			_y.zeros();
			return;
		}

		_y = z - H*_x;

		fmat PHT = _P*H.t();
		S = H*PHT + H*M + M.t()*H.t() + R;
		SI = inv(S);
		K = (PHT+M)*SI;

		_x = _x + K*_y;
		_P = _P - K*(H*_P + M.t());

		_z = z;
		x_post = _x;
		P_post = _P;
	}
	void get_prediction(const fcolvec& u, fcolvec& x, fmat& P)
	{
		if( (false == u.is_empty()) && (false == _B.is_empty()) )
			x = _F*_x + _B*u;
		else
			x = _F*_x;

		P = alpha_sq*_F*_P*_F.t() + _Q;
	}
	void get_update(const fcolvec& z, fcolvec& x, fmat& P)
	{
		if( true == z.is_empty() )
		{
			x = _x;
			P = _P;
			return;
		}

		fmat R = _R;
		fmat H = _H;
		P = _P;
		x = _x;

		fcolvec y = z - H*x;
		fmat PHT = P*H.t();

		fmat s = H*PHT + R;
		fmat k = PHT*inv(s);
		x = x + k*y;

		fmat i_kh = I - k*H;
		P = i_kh*P*i_kh.t() + k*R*k.t();
	}
	void residual(const fcolvec& z, fcolvec zo)
	{
		zo = z - _H*x_prior;
	}
	void measurement_of_state(const fcolvec& x, fcolvec& zo)
	{
		zo = _H*x;
	}
	float calc_log_likelihood()
	{
		return logpdf(_y, S);
	}
	float calc_mahalanobis()
	{
		mahalanobis = std::sqrt(arma::as_scalar(_y.t() * SI * _y));

		return mahalanobis;
	}
	float calc_mahalanobis(const arma::fcolvec& z)
	{
		const arma::fcolvec err = z - _H * _x;
		return std::sqrt(arma::as_scalar(err.t() * SI * err));
	}
	float alpha_sq;
	int dim_x;
	int dim_z;
	int dim_u;

	float likelihood, log_likelihood, mahalanobis;

	fcolvec _x, x_prior, x_post;
	fcolvec _y;
	fcolvec _z;
	fmat _P, P_prior, P_post;
	fmat _Q, _R, _H, _F, _B, M, K, S, SI, I;
} KFFilter_t;

typedef struct _track_t
{
	void init(int count, int filter_type)
	{
		x_.reserve(count);
		y_.reserve(count);
		h_.reserve(count);
		t_.reserve(count);

		float g, h, k;
//		g = 0.5f;

//		fx_.ons(g, h, k);

		float theta = 0.9f;
		fx_.dumping(theta, g, h, k);

		fx_.init(g, h, k, 1.0f);
		fy_.init(g, h, k, 1.0f);

		filter_type_ = filter_type;
	}
	void shrink()
	{
		x_.shrink_to_fit();
		y_.shrink_to_fit();
		h_.shrink_to_fit();
		t_.shrink_to_fit();
	}
	void update(const fvec& data)
	{
		x_.push_back(data(0));
		y_.push_back(data(1));

		if( data.size() > 2 )
			h_.push_back(data(2));

		if( data.size() > 3 )
			t_.push_back(data(3));

		if( xs_.size() > 2 )
		{
			xs_.push_back(fx_.update(1.0f, data(0), -1., -1., -1.));
			ys_.push_back(fy_.update(1.0f, data(1), -1., -1., -1.));
		}
		else
		{
			xs_.push_back(data(0));
			ys_.push_back(data(1));
			if( 1 == xs_.size() )
			{
				float dx = data(0) - xs_[0];
				float dy = data(1) - ys_[0];

				fx_.init(data(0), dx, 0.0f);
				fy_.init(data(1), dy, 0.0f);
			}
		}
	}
	void update(const float t, const fvec& data, fvec& sm)
	{
		switch( filter_type_ )
		{
			case filter_GH:
				sm[0] = fx_.update(t, data(0), -1., -1., -1.);
				sm[1] = fy_.update(t, data(1), -1., -1., -1.);
				break;

			case filter_KF:
				break;
		}
	}
	void update(const float t, const fvec& data)
	{
		x_.push_back(data(0));
		y_.push_back(data(1));

		if( data.size() > 2 )
			h_.push_back(data(2));

		if( xs_.size() > 2 )
		{
			fvec sm = zeros<fvec>(data.size());
			update(t, data, sm);
			xs_.push_back(sm[0]);
			ys_.push_back(sm[1]);
		}
		else
		{
			xs_.push_back(data(0));
			ys_.push_back(data(1));
			if( 1 == xs_.size() )
			{
				float dx = data(0) - xs_[0];
				float dy = data(1) - ys_[0];

				fx_.init(data(0), dx, 0.0f);
				fy_.init(data(1), dy, 0.0f);
			}
		}
		t_.push_back(t);
	}
	int filter_type_;
	std::vector<float> x_, y_, h_, t_;
	std::vector<float> xs_, ys_;
	GHFilter_t fx_, fy_;
} track_t;

typedef struct _rls_t
{
	void init(int filter_type)
	{
		pos_ = zeros<fvec>(3);
		to_ = 0.0f;
		vr_ = 0.0f;
		az_ = 0.0f;

		track_.init(1000, filter_type);
	}
	void set_az(const float az)
	{
		az_ = az / radian;
	}
	void set_pos(float x, float y, float h)
	{
		pos_(0) = x;
		pos_(1) = y;
		pos_(2) = h;
	}
	void set_rotate(float to)
	{
		to_ = to;
		vr_ = g360 / to_;
	}
	void do_rotate_pre(const float dt)
	{
		azn_ = az_ + vr_ * dt;
		if( azn_ >= g360 )
			azn_ -= g360;
	}
	void do_rotate_end()
	{
		az_ = azn_;
	}
	void set_cko(float r_cko, float f_cko)
	{
		r_cko_ = r_cko;
		f_cko_ = f_cko / radian;
	}
	float get_slant(const fvec& target)
	{
		float x = target(0) - pos_(0);
		float y = target(1) - pos_(1);
		float h = target(2) - pos_(2);
		float r = sqrt(x*x + y*y + h*h);

		return r;
	}
	void do_measure(const fvec& target, fvec& measure, fvec& noise)
	{
		fvec rel = zeros<fvec>(2);
		rel(0) = target(0) - pos_(0);
		rel(1) = target(1) - pos_(1);

		fvec rf = zeros<fvec>(2);
		get_rf(rel, rf);

		rf(0) = get_slant(target);

		fvec mea = zeros<fvec>(2);
		get_xy(rf, mea);
		measure(0) = pos_(0) + mea(0);
		measure(1) = pos_(1) + mea(1);

		rf(0) += r_cko_ * randn<float>();
		rf(1) += f_cko_ * randn<float>();

		fvec noi = zeros<fvec>(2);
		get_xy(rf, noi);
		noise(0) = pos_(0) + noi(0);
		noise(1) = pos_(1) + noi(1);
	}
	bool crossed_target(const fvec& pos)
	{
		fvec rf = zeros<fvec>(2);

		get_rf(pos, rf);

		return inside(az_, azn_, rf(1));
	}
	bool crossed_ray(const fvec& pos, const fvec& posn)
	{
		bool ret = false;

		fvec rf = zeros<fvec>(2);
		fvec rfn = zeros<fvec>(2);

		get_rf(pos, rf);
		get_rf(posn, rfn);

		bool crossed = north_crossed(rf(1), rfn(1));

		if( crossed )
		{
			if( rf(1) < rfn(1) )
				ret = inside(rf(1), rfn(1), az_);
		}
		else
		{
			if( rf(1) > rfn(1) )
				ret = inside(rf(1), rfn(1), az_);
		}

		return ret;
	}
	void print()
	{
		pos_.print();
	}
	fvec pos_;
	float to_, vr_;
	float az_, azn_;
	float r_cko_, f_cko_;
	track_t track_;
} rls_t;



typedef struct
{
	void init()
	{
		pos_ = zeros<fvec>(3);
		vf_ = zeros<fvec>(2);
		velo_ = zeros<fvec>(2);

		posn_ = zeros<fvec>(3);
		vfn_ = zeros<fvec>(2);
		velon_ = zeros<fvec>(2);

		th_ = 0.0f; ta_ = 0.0f; tf_ = 0.0f;
	}
	void set(const float x, const float y, const float h, const float v, const float f)
	{
		pos_(0) = x;
		pos_(1) = y;
		pos_(2) = h;

		vf_(0) = v;
		vf_(1) = f;

		get_xy(vf_, velo_);
	}
	void set(const fvec& pos, float v, float f)
	{
		pos_ = pos;

		vf_(0) = v;
		vf_(1) = f;

		get_xy(vf_, velo_);
	}
	void set_speed(const float accel, const float velo)
	{
		float dv = fabs(vf_(0) - velo);
		ta_ = fabs(dv / accel);

		accel_ = fabs(accel);
		if( velo < vf_(0) )
			accel_ = -accel_;
	}
	void set_height(const float climb, const float height)
	{
		float dh = fabs(pos_(2) - height);
		th_ = fabs(dh / climb);

		climb_ = fabs(climb);
		if( height < pos_(2) )
			climb_ = -climb_;
	}
	void set_f(const float gamma, const float f)
	{
		cmd_gamma_ = gamma;
		cmd_f_ = f;

		float gamma_sign;
		float df = fabs(vf_(1)*radian - f);
		if( df >= 180.0f )
		{
			df = 360.0f - df;
			gamma_sign = (f > vf_(1)*radian) ? -1.0f : 1.0f;
		}
		else
		{
			gamma_sign = (f < vf_(1)*radian) ? -1.0f : 1.0f;
		}

		tf_ = 2.0f*pi*vf_(0)*df/(9.81*tan(fabs(gamma)/radian)*360.0f);

		rotate_ = gamma_sign * df / (tf_ * radian);
	}
	void update_pre( float dt )
	{
		posn_(0) = pos_(0) + velo_(0)*dt;
		posn_(1) = pos_(1) + velo_(1)*dt;
		posn_(2) = pos_(2);

		if( th_ > 0.0f )
			posn_(2) += climb_*dt;

		vfn_ = vf_;
		if( ta_ > 0.0f )
			vfn_(0) = vf_(0) + accel_*dt;

		if( tf_ > 0.0f )
		{
			vfn_(1) = vf_(1) + rotate_*dt;
			if( vfn_(1) >= g360 )
				vfn_(1) -= g360;
			else if( vfn_(1) < 0.0f )
				vfn_(1) += g360;
		}

		if( (ta_ > 0.0f) || (tf_ > 0.0f) )
			get_xy(vfn_, velon_);
	}
	void update_end( float dt )
	{
		pos_ = posn_;
		vf_ = vfn_;

		if( (ta_ > 0.0f) && (tf_ > 0.0f) )
			set_f(cmd_gamma_, cmd_f_);

		get_xy(vf_, velo_);

		if( th_ > 0.0f )
			th_ = max(0.0f, th_ - dt);

		if( ta_ > 0.0f )
			ta_ = max(0.0f, ta_ - dt);

		if( tf_ > 0.0f )
			tf_ = max(0.0f, tf_ - dt);
	}
	fvec pos_;
	fvec vf_;
	fvec velo_;

	fvec posn_;
	fvec vfn_;
	fvec velon_;

	float th_, ta_, tf_;
	float accel_, rotate_, climb_;
	float cmd_f_, cmd_gamma_;
} target_t;

void time_command_1(const float ttx, target_t *ti)
{
	if( ttx == 20.0f )
		ti->set_speed(5.0f, 180.0f);

	if( ttx == 40.0f )
		ti->set_speed(2.0f, 200.0f);

	if( ttx == 10.0f )
		ti->set_height(12.0f, 8600.f);

	if( ttx == 200.0f )
		ti->set_f(30.0f, 345.0f);

	if( ttx == 210.0f )
		ti->set_speed(5.0f, 250.0f);

	if( ttx == 300.0f )
		ti->set_f(10.0f, 320.0f);

	if( ttx == 400.0f )
		ti->set_f(10.0f, 20.0f);

	if( ttx == 600.0f )
		ti->set_f(15.0f, 327.0f);
}

void time_command(const float ttx, target_t *ti)
{
//	time_command_1(ttx, ti);
}

#if 0
int main() 
{
	init_global();
	int ft = filter_GH;

	rls_t rls1;

	rls1.init(ft);
	rls1.set_pos(500.0f, 300.f, 20.0f);
	rls1.set_cko(60.0f, 0.1f/radian);
	rls1.set_rotate(5.0f);
	rls1.set_az(360.0f*randu());


	fvec target = zeros<fvec>(3);

	target(0) = 400000.0f;
	target(1) = 50000.0f;
	target(2) = 10000.0f;

	track_t track;

	track.init(10000, filter_GH);

	fvec posm = zeros<fvec>(2);
	fvec posn = zeros<fvec>(2);


	float v = 250.0f; // скорость
	float f = 265.0f; // курс

	float dt = 1.0f;

	fvec pos = zeros<fvec>(2);
	pos(0) = v;
	pos(1) = f / radian;

	std::vector<rls_t> rlist;
	rlist.push_back(rls1);

	target_t t1;
	t1.init();
	t1.set(target, pos(0), pos(1));

	std::vector<target_t> tlist;
	tlist.push_back(t1);

	float ttx = 0.0f;
	float ttx_max = 3500.f;

	track.update(ttx, t1.pos_);

	while( ttx <= ttx_max )
	{
		for( auto ti=tlist.begin(); ti != tlist.end(); ++ti )
		{
			time_command(ttx, &(*ti));

			ti->update_pre(dt);

			for( auto ri=rlist.begin(); ri != rlist.end(); ++ri )
			{
				if( ri->crossed_ray(ti->pos_, ti->posn_) )
				{
					ri->do_measure(ti->posn_, posm, posn);
					ri->track_.update(ttx+dt, posn);
				}
			}

			ti->update_end(dt);
		}

		for( auto ri=rlist.begin(); ri != rlist.end(); ++ri )
		{
			ri->do_rotate_pre(dt);

			for( auto ti=tlist.begin(); ti != tlist.end(); ++ti )
			{
				if( ri->crossed_target(ti->posn_) )
				{
					ri->do_measure(ti->posn_, posm, posn);
					ri->track_.update(ttx+dt, posn);
				}
			}

			ri->do_rotate_end();
		}

		ttx += dt;
		auto ti = tlist.begin();
		track.update(ttx, ti->pos_);
	}

	track.shrink();
	plt::plot(track.x_, track.y_);

	cout << track.x_.size();

	for( auto ri=rlist.begin(); ri != rlist.end(); ++ri )
	{
		ri->track_.shrink();
		cout << " " << ri->track_.xs_.size() << endl;
		plt::plot(ri->track_.x_, ri->track_.y_);
		plt::plot(ri->track_.xs_, ri->track_.ys_);
	}
//	plt::xlim(-400000.0f, 400000.0f);
//	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();

//	KFFilter_t kf;

//	kf.init(4, 2);

	fcolvec x = {1.f, 2.f, 3.f, 4.f};

	fmat pp = {
			{5., 6., 7., 8.},
			{9., 10., 11., 12.},
			{13., 14., 15., 16.},
			{17., 18., 19., 20.}
		};

	fmat ff = {
			{5.2, 6.2, 7.2, 8.2},
			{9.2, 10.2, 11.2, 12.2},
			{13.2, 14.2, 15.2, 16.2},
			{17.2, 18.2, 19.2, 20.2}
		};

	fmat qq = {
			{15., 16., 17., 18.},
			{19., 110., 111., 112.},
			{113., 114., 115., 116.},
			{117., 118., 119., 120.}
		};

//	pp = ff * pp * ff.t() + qq;

//	x = ff * x;
//	x.print();
//	fmat hh = {{1., 2., 3., 4.}, {5., 6., 7., 8.}};
//	cout << "rows " << hh.n_rows << " cols " << hh.n_cols << endl;
//	fcolvec cz = {12., 14.};
//	fcolvec	y = cz - hh * x;
//	y.print();
//
	fcolvec y = { 1., 2.};
	fmat si = {{ 3., 3.}, {3., 3}};

	fmat a = y.t()*si*y;
	float c = sqrt((float)a(0));

	a.print();
	cout << c << endl;
  return 0;
}
#endif

void fillx(fmat& amat, int dim)
{
//	amat.set_size(dim,dim);
//	amat.randu(dim,dim);
	amat = eye<fmat>(dim, dim);
}

void filly(fmat& amat, int dim, float dt)
{
//	amat.set_size(dim,dim);
//	amat.randu(dim,dim);
	amat = { {1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.} };
}

float power(float x, int n)
{
	float a = 1.0f;

	while( n )
	{
		if( n % 2 )
		{
			a *= x;
			--n;
		}
		else
		{
			x *= x;
			n >>= 1;
		}
	}

	return a;

}

void Q_continuous_white_noise(int dim, float dt, float den, int block_size, bool order, fmat& q)
{
	fmat a;
	int N = dim * block_size;

	if( (dim < 2) || (dim > 4) )
	{
		cout << "wrong size " << dim << endl;
	}

	if( 2 == dim )
	{
		a = { {power(dt,3)/3.0f, power(dt,2)/2.0f}, {power(dt,2)/2.0f, dt} };
		q = zeros<fmat>(N,N);

		q.submat(0,0,1,1) = a;
		if( block_size > 1 )
			q.submat(2,2,3,3) = a;
		if( block_size > 2 )
			q.submat(4,4,5,5) = a;
	}
	else if( 3 == dim )
	{
		a = { {power(dt,5)/20.0f, power(dt,4)/8.0f, power(dt,3)/6.0f}, 
		      {power(dt,4)/8.0f,  power(dt,3)/3.0f, power(dt,2)/2.0f}, 
		      {power(dt,3)/6.0f,  power(dt,2)/2.0f, dt} };

		q = zeros<fmat>(N,N);

		q.submat(0,0,2,2) = a;
		if( block_size > 1 )
			q.submat(3,3,5,5) = a;
		if( block_size > 2 )
			q.submat(6,6,8,8) = a;
	}
	else
	{
		a = { {power(dt,7)/252.f, power(dt,6)/72.f, power(dt,5)/30.0f, power(dt,4)/24.f}, 
		      {power(dt,6)/72.f, power(dt,5)/20.0f, power(dt,4)/8.0f, power(dt,3)/6.0f}, 
		      {power(dt,5)/30.0f, power(dt,4)/8.0f, power(dt,3)/3.0f, power(dt,2)/2.0f}, 
		      {power(dt,4)/24.0f, power(dt,3)/6.0f, power(dt,2)/2.0f, dt} };

		q = zeros<fmat>(N,N);

		q.submat(0,0,3,3) = a;
		if( block_size > 1 )
			q.submat(4,4,7,7) = a;
		if( block_size > 2 )
			q.submat(8,8,11,11) = a;
	}

	q *= den;
}
void Q_discrete_white_noise(int dim, float dt, float var, int block_size, bool order, fmat& q)
{
	fmat a;
	int N = dim * block_size;

	if( (dim < 2) || (dim > 4) )
	{
		cout << "wrong size " << dim << endl;
	}

	if( 2 == dim )
	{
		a = { {0.25f*power(dt,4), 0.5f*power(dt,3)}, {0.5f*power(dt,3), power(dt,2)} };
		q = zeros<fmat>(N,N);

		q.submat(0,0,1,1) = a;
		if( block_size > 1 )
			q.submat(2,2,3,3) = a;
		if( block_size > 2 )
			q.submat(4,4,5,5) = a;
	}
	else if( 3 == dim )
	{
		a = { {0.25f*power(dt,4), 0.5f*power(dt,3), 0.5f*power(dt,2)}, 
		      {0.5f*power(dt,3),  power(dt,2), dt}, 
		      {0.5f*power(dt,2),  dt, 1.0f} };

		q = zeros<fmat>(N,N);

		q.submat(0,0,2,2) = a;
		if( block_size > 1 )
			q.submat(3,3,5,5) = a;
		if( block_size > 2 )
			q.submat(6,6,8,8) = a;
	}
	else
	{
		a = { {power(dt,6)/36.f, power(dt,5)/12.f, power(dt,4)/6.0f, power(dt,3)/6.f}, 
		      {power(dt,5)/12.f, power(dt,4)/4.0f, power(dt,3)/2.0f, power(dt,2)/2.0f}, 
		      {power(dt,4)/6.0f, power(dt,3)/2.0f, power(dt,2), dt}, 
		      {power(dt,3)/6.0f, power(dt,2)/2.0f, dt, 1.0f} };

		q = zeros<fmat>(N,N);

		q.submat(0,0,3,3) = a;
		if( block_size > 1 )
			q.submat(4,4,7,7) = a;
		if( block_size > 2 )
			a.submat(8,8,11,11) = a;
	}

	q *= var;
}

void van_loan_diskretization(const fvec& F, const fvec& G, const float dt, fvec&sigma, fvec& Q)
{
}

typedef struct 
{
	void init(const float x0=0.0f, const float vel=1.0f, const float noise_scale=0.06f)
	{
		_x = x0;
		_vel = vel;
		_noise_scale = noise_scale;
	}
	void update(float &x, float &vel)
	{
		_vel += randn<float>() * _noise_scale;
		vel = _vel;

		_x += _vel;
		x = _x;
	}
	float sense(const float x, const float noise_scale=1.0f)
	{
		return x + randn<float>() * noise_scale;
	}
	float _x, _vel, _noise_scale;
} ConstantVelocityObject_t;

typedef struct 
{
	void init(const float x0=0.0f, const float vel=1.0f, const float acc=0.1f, const float acc_noise=0.1f)
	{
		_x = x0;
		_vel = vel;
		_acc = acc;
		_acc_noise_scale = acc_noise;
	}
	void update(float &x, float &vel)
	{
		_acc += std::abs(randn<float>()) * _acc_noise_scale;
		_vel += _acc;
		vel = _vel;

		_x += _vel;
		x = _x;
	}
	float sense(const float x, const float noise_scale=1.0f)
	{
		return x + randn<float>() * noise_scale;
	}
	float _x, _vel, _acc, _acc_noise_scale;
} ConstantAccelerationObject_t;

#if 0
int main()
{
	fmat A, B, F, x, y, z;
	fmat G;

	float dt = 0.1f;
	F = {{0.0f, 1.0f}, {-1.0f, 0.0f}};
	G = {0.0f, 2.0f};
	
	x = zeros<fmat>(G.size(), G.size());
	x.tail_rows(1) = G;



	y = zeros<fmat>(G.size(), G.size());
	y.tail_cols(1) = G.t();


	z = x*y;
//	z.print();

	int n = F.n_rows;
	A = zeros<fmat>(2*n,2*n);

	A.submat(0,0,n-1,n-1) = -F*dt;
	A.submat(0,n, n-1,2*n-1) = z*dt;
	A.submat(n,n, 2*n-1, 2*n-1) = F.t()*dt;

	A.print();

	B = expmat(A);

	B.print();

	return 0;
}
#endif

typedef struct
{
	void init(const float R, const float Q, const float P=20.0f)
	{
		kf.init(1, 1);
		kf._x = {0.0f};
		kf._R *= R;
		kf._Q *= Q;
		kf._P *= P;
		kf._F = eye<fmat>(1, 1);
		kf._H = eye<fmat>(1, 1);
	}
	void predict(const fcolvec& u, const fmat& b, const fmat& f, const fmat& q)
	{
		kf.predict(u, b, f, q);
	}
	void update(const fcolvec& z, const fmat& r, const fmat& h)
	{
		kf.update(z, r, h);
	}
	KFFilter_t kf;
} ZeroOrderKF_t;

typedef struct
{
	void init(const float R, const float Q, const float dt)
	{
		kf.init(2, 1);
		kf._x = zeros<fcolvec>(2);
		kf._R *= R;
		Q_discrete_white_noise(2, dt, Q, 1, false, kf._Q);
		kf._P *= {{100.0f, 0.0f}, {0.0f, 1.0f}};
		kf._F = {{1.0f, dt}, {0.0f, 1.0f}};
		kf._H = {{1.0f, 0.0f}};
	}
	void predict(const fcolvec& u, const fmat& b, const fmat& f, const fmat& q)
	{
		kf.predict(u, b, f, q);
	}
	void update(const fcolvec& z, const fmat& r, const fmat& h)
	{
		kf.update(z, r, h);
	}
	KFFilter_t kf;
} FirstOrderKF_t;

typedef struct
{
	void init(const float R_std, const float Q, const float dt, const float P=100.0f)
	{
		kf.init(3, 1);
		kf._x = zeros<fcolvec>(3);
		kf._R *= R_std*R_std;
		Q_discrete_white_noise(3, dt, Q, 1, false, kf._Q);
		kf._P[0, 0] = P;
	        kf._P[1, 1] = 1.0f;
		kf._P[2, 2] = 1.0f;
		kf._F = {{1.0f, dt, 0.5f*dt*dt}, 
			{0.0f, 1.0f, dt}, 
			{0.0f, 0.0f, 1.0f}};
		kf._H = {{1.0f, 0.0f, 0.0f}};
	}
	void predict(const fcolvec& u, const fmat& b, const fmat& f, const fmat& q)
	{
		kf.predict(u, b, f, q);
	}
	void update(const fcolvec& z, const fmat& r, const fmat& h)
	{
		kf.update(z, r, h);
	}
	float calc_mahalanobis(const fcolvec& z)
	{
		return kf.calc_mahalanobis(z);
	}
	KFFilter_t kf;
} SecondOrderKF_t;

#if 0
int main()
{
#if 0	
	float dt = 0.1f;
	fmat F = {{1.0f, dt}, {0.0f, 1.0f}};
	fcolvec x = {10.0f, 4.5f};
	fmat P = diagmat((fmat){500.0f, 49.0f});
	fmat Q = zeros<fmat>(2, 2);

	F.print();
	x.print();
	P.print();
	Q.print();

	KFFilter_t kf;

	kf.init(2, 2);

	kf._x = x;
	kf._P = P;
	kf._F = F;
	kf._Q = Q;

	fmat zero;

	for( auto i=0; i<5; ++i)
	{
		kf.predict(zero, zero, zero, zero);
		kf._x.print();
	}

	kf._P.print();

	fmat q;

	Q_discrete_white_noise(2, 1.0f, 2.35f, 1, false, q);
	q.print();

	fmat qx;

	Q_continuous_white_noise(3, 1.0f, 1.0f, 1, false, qx);
	qx.print();

	fmat R = {5.0f};
	R.print();
	fmat H = {1.0f, 0.0f};
	H.print();
#endif

	ConstantVelocityObject_t obj;

	obj.init();

	std::vector<float> xs, zs;

	float x, v, z;
	for( auto i=0; i<50; ++i )
	{
		obj.update(x, v);
		xs.push_back(x);
		z = obj.sense(x);
		zs.push_back(z);
	}

	plt::plot(xs);
	plt::plot(zs);

//	plt::xlim(-400000.0f, 400000.0f);
//	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();

	return 0;
}
#endif

#if 0
int main()
{
	fcolvec zerov;
	fmat zerom;
	float R = 1.0f, Q = 0.0001f, dt = 1.0f;

	ConstantVelocityObject_t obj;
	obj.init(0.0f, 0.5f, Q);

	ZeroOrderKF_t kf0;
	kf0.init(R, Q, dt);

	FirstOrderKF_t kf1;
	kf1.init(R, Q, dt);

	SecondOrderKF_t kf2;
	kf2.init(R, 0, dt);

	std::vector<float> xs, zs, f0, f1, f2;

	float x, v, z;
	int count = 200;
	for( auto i=0; i<count; ++i )
	{
		obj.update(x, v);
		xs.push_back(x);
		z = obj.sense(x);
		zs.push_back(z);

#if 0		
		kf0.predict(zerov, zerom, zerom, zerom);
		fcolvec zv = {z};
		kf0.update(zv, zerom, zerom);
		f0.push_back(kf0.kf._x[0]);
#endif		
#if 0		
		kf1.predict(zerov, zerom, zerom, zerom);
		fcolvec zv = {z};
		kf1.update(zv, zerom, zerom);
		f1.push_back(kf1.kf._x[0]);
#endif		
		kf2.predict(zerov, zerom, zerom, zerom);
		fcolvec zv = {z};
		kf2.update(zv, zerom, zerom);
		f2.push_back(kf2.kf._x[0]);
	}

	plt::plot(xs);
	plt::plot(zs);
	plt::plot(f2);

//	plt::xlim(-400000.0f, 400000.0f);
//	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();

	return 0;
}
#endif

int main()
{
	fcolvec zerov;
	fmat zerom;
	float R = 6.0f, Q = 0.02f, dt = 1.0f;

	ConstantAccelerationObject_t obj;
	obj.init(0.0f, 1.0f, 0.1f, Q);

	ZeroOrderKF_t kf0;
	kf0.init(R, Q, dt);

	FirstOrderKF_t kf1;
	kf1.init(R, Q, dt);

	SecondOrderKF_t kf2;
	kf2.init(R, Q, dt);

	std::vector<float> xs, zs, f0, f1, f2, ma, ma1, resi;

	float x, v, z;
	int count = 80;
	for( auto i=0; i<count; ++i )
	{
		obj.update(x, v);
		xs.push_back(x);
		z = obj.sense(x, R);
		zs.push_back(z);

#if 0		
		kf0.predict(zerov, zerom, zerom, zerom);
		fcolvec zv = {z};
		kf0.update(zv, zerom, zerom);
		f0.push_back(kf0.kf._x[0]);
#endif		
#if 0		
		kf1.predict(zerov, zerom, zerom, zerom);
		fcolvec zv = {z};
		kf1.update(zv, zerom, zerom);
		f1.push_back(kf1.kf._x[0]);
#endif		
		kf2.predict(zerov, zerom, zerom, zerom);
		fcolvec zv = {z};
		ma1.push_back(kf2.kf.calc_mahalanobis(zv));
		kf2.update(zv, zerom, zerom);
		f2.push_back(kf2.kf._x[0]);
		ma.push_back(kf2.kf.mahalanobis);
		resi.push_back(kf2.kf._y[0]);
	}

		xs.push_back(x);
		z = obj.sense(x, R);
		zs.push_back(z);

		kf2.predict(zerov, zerom, zerom, zerom);
		fcolvec zv = {z};
		ma1.push_back(kf2.kf.calc_mahalanobis(zv));
		kf2.update(zv, zerom, zerom);
		f2.push_back(kf2.kf._x[0]);
		ma.push_back(kf2.kf.mahalanobis);
		resi.push_back(kf2.kf._y[0]);

	plt::plot(xs);
	plt::plot(zs);
	plt::plot(f2);

//	plt::xlim(-400000.0f, 400000.0f);
//	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();

//	vec mx = {1.0f, 2.0f}, mmu = {1.1f, 3.5f};
//	mat msi = {{1.0f, 0.1f}, {0.1f, 13.0f}};
//	vec mx = {3.0f}, mmu = {3.5f};
//	mat msi = {4.0f*4.0f};

//	double mma = mahalanobis(mx, mmu, msi);
//	std::cout << mma << endl;

	plt::plot(ma);
	plt::plot(ma1);

//	plt::xlim(-400000.0f, 400000.0f);
//	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();

	plt::plot(resi);

//	plt::xlim(-400000.0f, 400000.0f);
//	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();
	return 0;
}

