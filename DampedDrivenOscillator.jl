# the damped driven oscillator.
# for P2600

# usage: set parameters in the initialize!() function call at the 
# bottom of the script. this constructs a "manifold" type with the 
# relevant SHO parameters.

# you can run the simulation using either the ODE solver from the 
# DiffEqs package or by numerical integration.

# the function call run_renderer() will evolve the system and produce
# a real time rendering of the graphs with GLMakie. 

using Plots
using DifferentialEquations
using GLMakie

mutable struct Particle
	x::Vector{Float64}	# position
	v::Vector{Float64}	# velocity
	E::Vector{Float64}	# energy
end

# struct for the SHO "manifold"
mutable struct Manifold
	dt::Float64		# time step
	tt::Float64		# total time 
	nt::Int64		# num time steps
	t::Float64		# current time (manifold time)
	ω0::Float64		# undamped angular frequency
	b::Float64		# viscous damping coefficient
	c::Float64		# anharmonic
	ω::Float64		# driving frequency
	A::Float64		# driving amplitude
	particle::Particle	# particle on the manifold
end

# returns a manifold type with simulation parameters specified
function initialize!(;dt::Float64, tt::Float64, ω0::Float64, ω::Float64, 
		b::Float64, c::Float64, A::Float64, x0::Float64, v0::Float64)
	nt = Int64(tt/dt)
	particle = Particle(zeros(nt), zeros(nt), zeros(nt))
	particle.x[1] = x0
	particle.v[1] = v0
	t = 0.0
	return Manifold(dt, tt, nt, t, ω0, b, c, ω, A, particle)
end

function get_acceleration(m::Manifold, t::Int64)
	a = -(m.ω0)^2 * m.particle.x[t] - 
		m.b * m.particle.v[t] - 
		m.c * m.particle.x[t]^3 + 
		m.A * cos(m.ω * m.t)
	return a
end

function get_energy(m::Manifold, t::Int64)
	PE = 0.5 * (m.ω0)^2 * m.particle.x[t]^2 + 
		0.25 * m.c * m.particle.x[t]^4
	KE = 0.5 * m.particle.v[t]^2
	return EK + EP
end

# euler integrator
function integrate_euler!(m::Manifold, accel, t::Int64)
	m.particle.v[t+1] = m.particle.v[t] + accel * m.dt
	m.particle.x[t+1] = m.particle.x[t] + m.particle.v[t+1] * m.dt
	m.t += m.dt
end

# velocity verlet integrator
function integrate_verlet!(m::Manifold, accel::Float64, t::Int64)
	# admittedly, this is a terrible place to put an if statement
	# should change this so the first verlet step is outside the loop
	if t == 1
		m.particle.v[t+1] = m.particle.v[t] + accel * m.dt * 0.5
		m.particle.x[t+1] = m.particle.x[t] + m.particle.v[t+1] * m.dt * 0.5
	else
		m.particle.v[t+1] = m.particle.v[t-1] + accel * m.dt
		m.particle.x[t+1] = m.particle.x[t] + m.particle.v[t+1] * m.dt
	end
	m.t += m.dt
end

# numerically integrate the ode from t = 1:tt
function evolve!(m::Manifold)
	for t in 1:(m.nt-1)
		# energy
		m.particle.E[t] = get_energy(m,t)

		# acceleration
		accel = get_acceleration(m,t) 

		# integrate using velocity verlet or euler
		integrate_verlet!(m, accel, t)
		#integrate_euler!(m, accel, t)

		t += 1
	end
end

function render!(m::Manifold)

	# glmakie code for the renderer
	# this could use a clean up
	render_xt = Observable(Point2f[(m.t[1], m.particle.x[1])])
	render_xv = Observable(Point2f[(m.particle.x[1], m.particle.v[1])])
	render_E = Observable(Point2f[(m.t[1], m.particle.E[1])])
	fontsize_theme = Theme(fontsize=40)
	set_theme!(fontsize_theme)
	fig = Figure(resolution=(600,400))
	ax1 = Axis(fig[1,1], xlabel = "t", ylabel = "x")
	ax2 = Axis(fig[1,2], xlabel = "x", ylabel = "v")
	ax3 = Axis(fig[2,1], xlabel = "t", ylabel = "E")
	lines!(ax1, render_xt, linewidth=5)
	lines!(ax2, render_xv, linewidth=5)
	lines!(ax3, render_E, linewidth=5)
	limits!(ax1, 0, m.tt, -m.particle.x[1] * 1.2, m.particle.x[1] * 1.2)
	limits!(ax2, -4, 4, -4, 4)
	limits!(ax3, 0, m.tt, -2, 2)
	display(fig)

	for t in 1:(m.nt-1)
		# update render
		render_xt[] = push!(render_xt[], Point2f(m.t, m.particle.x[t]))
		render_xv[] = push!(render_xv[], Point2f(m.particle.x[t], m.particle.v[t]))
		render_E[] = push!(render_E[], Point2f(m.t, m.particle.E[t]))
		yield()

		# energy
		m.particle.E[t] = get_energy(m,t)

		# acceleration
		accel = get_acceleration(m,t) 

		# integrate using velocity verlet or euler
		integrate_verlet!(m, accel, t)
		#integrate_euler!(m, accel, t)

		t += 1
	end
end

# the "tendency" function, for the DiffEqs package solver
# Γ is a vector containing the phase space variables, Γ = [x,v]
function tendency!(dΓ::Vector{Float64}, Γ::Vector{Float64}, p, t::Float64)
	x = Γ[1]
	v = Γ[2]
	k = (m.ω0)^2
	a = -k * x - m.b * v - 0.0 * x^3 + m.A * cos(m.ω * t)
	dΓ[1] = v
	dΓ[2] = a
end

function run_odesolver()
	Γ0 = [m.particle.x[1], m.particle.v[1]]
	tspan = (0.0, m.tt)
	prob = ODEProblem(tendency!, Γ0, tspan)
	sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
	
	len = length(sol.t)
	xdata = [sol.u[i][1] for i in 1:len]
	vdata = [sol.u[i][2] for i in 1:len]
	tdata = sol.t
	make_plots(xdata, vdata, tdata)
end

function run_manual()
	evolve!(m)
	
	tpoints::Vector{Float64} = 0.0:m.dt:((m.tt)-(m.dt))
	make_plots(m.particle.x, m.particle.v, tpoints)
end

function run_renderer()
	render!(m)
end

function make_plots(x::Vector{Float64}, v::Vector{Float64}, t::Vector{Float64})
	xt_plot = Plots.plot(t, x, xlabel = "t", ylabel = "x(t)")
	xv_plot = Plots.plot(x, v, xlabel = "x", ylabel = "v")
	Plots.plot(xt_plot, xv_plot, size=(600,300))
end

# initialize manifold m 
m = initialize!(
	dt = 0.01,	# time step	
	tt = 100.0,	# total time
	ω0 = 1.0,	# undamped frequency
	b = 0.1,	# damping coefficient 
	c = 1.0,	# anharmonic
	ω = 1.2,	# driving frequency
	A = 0.7,	# driving amplitude
	x0 = -3.0,	# initial position of particle
	v0 = 0.0	# initial velocity of particle
	)

#run_odesolver()		# run the simulation using the DiffEqs ODE solver
#run_manual()		# or run "manually" using the one of the numerical integrators
run_renderer()		# additionally render the graphs in real time (slows everything down)
