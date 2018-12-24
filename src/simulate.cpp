#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <armadillo> 
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace arma;
using namespace std;

float pi;
float g22, g45, g67, g90, g112, g135, g157, g180, g202, g225, g247, g270, g292, g315, g337, g360, radian;

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

void get_xy(float r, float f, float &x, float &y)
{
	if( r < 0.0f || f < 0.0f )
	{
		x = 0.0f;
		y = 0.0f;
	}
	if( f <= g90 )
	{
		x = r * sin(f);
		y = r * cos(f);
	}
	else if( f <= g180 )
	{
		f -= g90;
		x = r * cos(f);
		y = -r * sin(f);
	}
	else if( f <= g270 )
	{
		f -= g180;
		x = -r * sin(f);
		y = -r * cos(f);
	}
	else if( f <= g360 )
	{
		f -= g270;
		x = -r * cos(f);
		y = r * sin(f);
	}	
	else
	{
		f -= g360;
		x = r * sin(f);
		y = r * cos(f);
	}
}

float get_slant(float x, float y, float h)
{
	float r = sqrt(x*x + y*y + h*h);

	return r;
}

void get_from_slant(float r, float h, float f, float &x, float &y)
{
	float rm = sqrt(r*r - h*h);

	get_xy(rm, f, x, y);
}


void get_rf(float x, float y, float &r, float &f)
{
	r = std::hypot(x, y);

	if( r > 0.001f )
	{
		f = acos(y / r);
		if( x < 0.0f )
			f = g360 - f;
	}
	else
		f = 0.0f;
}

void get_rls_from_sim(float rx, float ry, float x, float y, float h, float &xs, float &ys, float &xn, float &yn)
{
	float xrel = x - rx;
	float yrel = y - ry;

       	float r, f;
	get_rf(xrel, yrel, r, f);
	float rs = get_slant(xrel, yrel, h);

	float rn = rs + 150.0f * randn<float>();
	float fn = f + 0.1/radian*randn<float>();

	get_xy(rs, f, xs, ys);
	xs += rx;
	ys += ry;

	get_xy(rn, fn, xn, yn);
	xn += rx;
	yn += ry;
}

void lt()
{
	std::vector<float> xreal, yreal;
	std::vector<float> xr1, yr1, xr1n, yr1n;

	xreal.reserve(1000);
	yreal.reserve(1000);

	xr1.reserve(1000);
	yr1.reserve(1000);
	xr1n.reserve(1000);
	yr1n.reserve(1000);

	float xpos = 400000.0f;
	float ypos = 50000.0f;

	float v = 250.0f; // скорость
	float f = 265.0f; // курс

	float dt = 5.0f;

	float vx, vy;

	get_xy(v, f / radian, vx, vy);
	std::cout << "vx = " << vx << " vy " << vy << std::endl;
	
	xreal.push_back(xpos);
	yreal.push_back(ypos);

	float rx = 0.0f;
	float ry = 0.0f;

	float h = 10000.0f;
	float xs, ys, xn, yn;

	get_rls_from_sim(rx, ry, xpos, ypos, h, xs, ys, xn, yn);
	xr1.push_back(xs);
	yr1.push_back(ys);
	xr1n.push_back(xn);
	yr1n.push_back(yn);


	for( int i=0; i<700; ++i )
	{
		xpos += vx * dt;
		xreal.push_back(xpos);

		ypos += vy * dt;
		yreal.push_back(ypos);
	
		get_rls_from_sim(rx, ry, xpos, ypos, h, xs, ys, xn, yn);
		xr1.push_back(xs);
		yr1.push_back(ys);
		xr1n.push_back(xn);
		yr1n.push_back(yn);
	}

	xreal.shrink_to_fit();
	yreal.shrink_to_fit();

	xr1.shrink_to_fit();
	yr1.shrink_to_fit();
	xr1n.shrink_to_fit();
	yr1n.shrink_to_fit();

	plt::plot(xreal, yreal);
	plt::plot(xr1, yr1);
	plt::plot(xr1n, yr1n);
	plt::xlim(-400000.0f, 400000.0f);
	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();
}

typedef struct _rls_t
{
	void init()
	{
		pos_ = zeros<fvec>(3);
		to_ = 0.0f;
		vr_ = 0.0f;
		az_beg_ = 0.0f;
		az_end_ = 0.0f;
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
	void do_rotate(float t)
	{
		az_end_ = az_beg_ + vr_ * t;
		if( az_end_ >= g360 )
			az_end_ -= g360;

		// ...
		//

		az_beg_ = az_end_;

	}
	void set_cko(float r_cko, float f_cko)
	{
		r_cko_ = r_cko;
		f_cko_ = f_cko / radian;
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
	void get_rf(const fvec& target, fvec& rf)
	{
		float x = target(0);
		float y = target(1);
		rf(0) = std::hypot(x, y);

		if( rf(0) > 0.001f )
		{
			rf(1) = acos(y / rf(0));
			if( x < 0.0f )
				rf(1) = g360 - rf(1);
		}
		else
			rf(1) = 0.0f;
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
	void print()
	{
		pos_.print();
	}
	fvec pos_;
	float to_, vr_;
	float az_beg_, az_end_;
	float r_cko_, f_cko_;
} rls_t;

void target_update(const fvec& target)
{
}

typedef struct _traj_t
{
	void init(int count)
	{
		x.reserve(count);
		y.reserve(count);
	}
	void shrink()
	{
		x.shrink_to_fit();
		y.shrink_to_fit();
	}
	void update(const fvec& data)
	{
		x.push_back(data(0));
		y.push_back(data(1));
	}
	std::vector<float> x, y;
} traj_t;

int main() 
{
	init_global();

	rls_t rls1, rls2, rls3, rls4, rls5, rls6;

	rls1.init();
	rls1.set_pos(500.0f, 300.f, 20.0f);
	rls1.set_cko(60.0f, 0.1f/radian);

	rls2.init();
	rls2.set_pos(3000.0f, -2000.0f, 100.0f);
	rls2.set_cko(300.0f, 0.2f/radian);

	rls3.init();
	rls3.set_pos(100000.0f, 100000.0f, 200.0f);
	rls3.set_cko(150.0f, 0.15f/radian);

	rls4.init();
	rls4.set_pos(100000.0f, -100000.0f, 700.0f);
	rls4.set_cko(300.0f, 0.2f/radian);

	rls5.init();
	rls5.set_pos(-100000.0f, -100000.0f, 100.0f);
	rls5.set_cko(150.0f, 0.15f/radian);

	rls6.init();
	rls6.set_pos(-100000.0f, 100000.0f, 70.0f);
	rls6.set_cko(300.0f, 0.2f/radian);

	fvec target = zeros<fvec>(3);

	target(0) = 400000.0f;
	target(1) = 50000.0f;
	target(2) = 10000.0f;

	fvec measure = zeros<fvec>(2);
	fvec noise = zeros<fvec>(2);

	traj_t tt, r1m, r1n, r2m, r2n, r3m, r3n, r4m, r4n;
	traj_t r5m, r5n, r6m, r6n;
	tt.init(1000);
	r1m.init(1000);
	r1n.init(1000);
	r2m.init(1000);
	r2n.init(1000);
	r3m.init(1000);
	r3n.init(1000);
	r4m.init(1000);
	r4n.init(1000);
	r5m.init(1000);
	r5n.init(1000);
	r6m.init(1000);
	r6n.init(1000);

	fvec posm = zeros<fvec>(2);
	fvec posn = zeros<fvec>(2);


	float v = 250.0f; // скорость
	float f = 265.0f; // курс

	float dt = 5.0f;

	fvec pos = zeros<fvec>(2);
	pos(0) = v;
	pos(1) = f / radian;

	fvec vv = zeros<fvec>(2);

	rls1.get_xy(pos, vv);

	std::cout << "vx = " << vv(0) << " vy " << vv(1) << std::endl;

	int count = 0;
	tt.update(target);
	count++;

	rls1.do_measure(target, posm, posn);	
	r1m.update(posm);
	r1n.update(posn);
	
	for( int i=0; i<150; ++i )
	{
		target(0) += vv(0) * dt;
		target(1) += vv(1) * dt;
		tt.update(target);
		count++;

		rls1.do_measure(target, posm, posn);	
		r1m.update(posm);
		r1n.update(posn);

		if( 0 == count % 2 )
		{
			rls3.do_measure(target, posm, posn);	
			r3m.update(posm);
			r3n.update(posn);
			
			rls5.do_measure(target, posm, posn);	
			r5m.update(posm);
			r5n.update(posn);
		}

		if( 0 == count % 4 )
		{
			rls2.do_measure(target, posm, posn);	
			r2m.update(posm);
			r2n.update(posn);
			
			rls4.do_measure(target, posm, posn);	
			r4m.update(posm);
			r4n.update(posn);
			
			rls6.do_measure(target, posm, posn);	
			r6m.update(posm);
			r6n.update(posn);
		}

		target(0) += vv(0) * dt;
		target(1) += vv(1) * dt;
		tt.update(target);
		count++;

		rls1.do_measure(target, posm, posn);	
		r1m.update(posm);
		r1n.update(posn);

		if( 0 == count % 2 )
		{
			rls3.do_measure(target, posm, posn);	
			r3m.update(posm);
			r3n.update(posn);
			
			rls5.do_measure(target, posm, posn);	
			r5m.update(posm);
			r5n.update(posn);
		}

		if( 0 == count % 4 )
		{
			rls2.do_measure(target, posm, posn);	
			r2m.update(posm);
			r2n.update(posn);
			
			rls4.do_measure(target, posm, posn);	
			r4m.update(posm);
			r4n.update(posn);
			
			rls6.do_measure(target, posm, posn);	
			r6m.update(posm);
			r6n.update(posn);
		}


		target(0) += vv(0) * dt;
		target(1) += vv(1) * dt;
		tt.update(target);
		count++;

		rls1.do_measure(target, posm, posn);	
		r1m.update(posm);
		r1n.update(posn);

		if( 0 == count % 2 )
		{
			rls3.do_measure(target, posm, posn);	
			r3m.update(posm);
			r3n.update(posn);
			
			rls5.do_measure(target, posm, posn);	
			r5m.update(posm);
			r5n.update(posn);
		}

		if( 0 == count % 4 )
		{
			rls2.do_measure(target, posm, posn);	
			r2m.update(posm);
			r2n.update(posn);
			
			rls4.do_measure(target, posm, posn);	
			r4m.update(posm);
			r4n.update(posn);
			
			rls6.do_measure(target, posm, posn);	
			r6m.update(posm);
			r6n.update(posn);
		}


		target(0) += vv(0) * dt;
		target(1) += vv(1) * dt;
		tt.update(target);
		count++;

		rls1.do_measure(target, posm, posn);	
		r1m.update(posm);
		r1n.update(posn);

		if( 0 == count % 2 )
		{
			rls3.do_measure(target, posm, posn);	
			r3m.update(posm);
			r3n.update(posn);
			
			rls5.do_measure(target, posm, posn);	
			r5m.update(posm);
			r5n.update(posn);
		}

		if( 0 == count % 4 )
		{
			rls2.do_measure(target, posm, posn);	
			r2m.update(posm);
			r2n.update(posn);
			
			rls4.do_measure(target, posm, posn);	
			r4m.update(posm);
			r4n.update(posn);
			
			rls6.do_measure(target, posm, posn);	
			r6m.update(posm);
			r6n.update(posn);
		}


		target(0) += vv(0) * dt;
		target(1) += vv(1) * dt;
		tt.update(target);
		count++;

		rls1.do_measure(target, posm, posn);	
		r1m.update(posm);
		r1n.update(posn);

		if( 0 == count % 2 )
		{
			rls3.do_measure(target, posm, posn);	
			r3m.update(posm);
			r3n.update(posn);
			
			rls5.do_measure(target, posm, posn);	
			r5m.update(posm);
			r5n.update(posn);
		}

		if( 0 == count % 4 )
		{
			rls2.do_measure(target, posm, posn);	
			r2m.update(posm);
			r2n.update(posn);
			
			rls4.do_measure(target, posm, posn);	
			r4m.update(posm);
			r4n.update(posn);
			
			rls6.do_measure(target, posm, posn);	
			r6m.update(posm);
			r6n.update(posn);
		}
	}

	tt.shrink();
	r1m.shrink();
	r1n.shrink();
	r2m.shrink();
	r2n.shrink();
	r3m.shrink();
	r3n.shrink();
	r4m.shrink();
	r4n.shrink();
	r5m.shrink();
	r5n.shrink();
	r6m.shrink();
	r6n.shrink();

	plt::plot(tt.x, tt.y);
//	plt::plot(r1m.x, r1m.y);
	plt::plot(r1n.x, r1n.y);
//	plt::plot(r2m.x, r2m.y);
	plt::plot(r2n.x, r2n.y);
//	plt::plot(r3m.x, r3m.y);
	plt::plot(r3n.x, r3n.y);
//	plt::plot(r4m.x, r4m.y);
	plt::plot(r4n.x, r4n.y);
//	plt::plot(r5m.x, r5m.y);
	plt::plot(r5n.x, r5n.y);
//	plt::plot(r6m.x, r6m.y);
	plt::plot(r6n.x, r6n.y);
	plt::xlim(-400000.0f, 400000.0f);
	plt::ylim(-100000.0f, 100000.0f);
	plt::grid(true);
	plt::save("./simulate.png");
	plt::show();
//#endif	
  return 0;
}
