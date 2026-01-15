# Numerical Projectile Motion Simulation: RK4 vs Euler Comparison
A Python-based numerical simulation of projectile motion with interactive visualization comapring RK4 and Euler time-step integrations. Parameters such as mass, drag, initial velocity, initial height, and more can be interacted with.

---

## Overview

This project numerically simulates two-dimensional projectile motion **with air resistance enabled**, focusing on a direct comparison between numerical integration methods.

The same physical system (same drag model, timestep, and initial conditions) is solved using:

* **Fourth-order Runge–Kutta (RK4)**
* **Forward Euler integration**

This allows clear visualization of numerical accuracy, stability, and error accumulation between integrators when applied to a non-linear dynamical system with drag.

---

## Features

* Projectile motion with air resistance (always enabled)
* Air resistance models:
  * Quadratic drag (default)
  * Linear drag (optional via parameter toggle)
* Fixed physical parameters for controlled comparison
* Numerical integration methods:
  * Fourth-order Runge–Kutta (RK4)
  * Forward Euler
* Direct integrator comparison:
  * Identical initial conditions and timestep
  * Identical force model
* Automatic ground-impact detection with linear interpolation
* Static visualizations:
  * Trajectory comparison (RK4 vs Euler)
  * Speed vs time (RK4 vs Euler)
  * Height vs time (RK4 vs Euler)
* Animated visualization:
  * Overlayed 2D projectile motion
  * RK4 and Euler shown simultaneously
  * Fading trajectory trails
  * Time overlay formatted as **mm:ss.hh**
  * Color-coded integrator key

---

## Methods

* Newton’s Second Law applied to two-dimensional motion with drag
* Coupled first-order ordinary differential equations
* Numerical time integration using:
  * Fourth-order Runge–Kutta (RK4)
  * Forward Euler method
* Fixed timestep integration
* Non-linear drag force (quadratic or linear)
* Ground-impact event detection using linear interpolation
* Side-by-side numerical comparison of integration accuracy

---

## Requirements

* Python 3
* NumPy
* Matplotlib
* ipywidgets (for interactive controls)
* Jupyter Notebook or Google Colab (recommended)

---

## Usage

1. Open the notebook in Jupyter or Google Colab.
2. Adjust physical parameters (launch angle, speed, drag model, timestep).
3. Run the notebook to:
   * Compare RK4 and Euler trajectories under identical conditions
   * Observe numerical error accumulation
   * View static plots and a combined overlay animation

---

## Possible Extensions

* Adaptive timestep methods
* Energy conservation and numerical error analysis
* Additional integrators (Verlet, midpoint, symplectic methods)
* Quantitative comparison of Euler vs RK4 error
* Experimental validation with real projectile data

---

## Motivation

Projectile motion with air resistance has no closed-form analytical solution.  
This makes it an ideal test case for comparing numerical integration methods.

By keeping the physical system fixed and varying only the integrator, this project highlights:
* Stability differences
* Error accumulation
* The limitations of first-order methods
* The advantages of higher-order schemes for non-linear dynamics
