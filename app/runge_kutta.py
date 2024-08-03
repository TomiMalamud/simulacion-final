from flask import Blueprint, render_template, request
import numpy as np

ode_solver_bp = Blueprint('ode_solver', __name__)

def solve_ode(t_max, step_size):
    def f(t, y, dy):
        return 4 * dy * dy + 6 * y + 8 * t
    
    t = 0
    y = 0
    dy = 0.05
    results = []

    while t <= t_max:
        y_prime2 = f(t, y, dy)
        h = step_size
        
        k1 = h * dy
        l1 = h * y_prime2
        
        k2 = h * (dy + 0.5 * l1)
        l2 = h * f(t + 0.5 * h, y + 0.5 * k1, dy + 0.5 * l1)
        
        k3 = h * (dy + 0.5 * l2)
        l3 = h * f(t + 0.5 * h, y + 0.5 * k2, dy + 0.5 * l2)
        
        k4 = h * (dy + l3)
        l4 = h * f(t + h, y + k3, dy + l3)

        results.append({
            'yPrime2': f"{y_prime2:.4f}",
            't': f"{t:.4f}",
            'y': f"{y:.4f}",
            'z': f"{dy:.4f}",
            'k1': f"{k1:.4f}",
            'l1': f"{l1:.4f}",
            'k2': f"{k2:.4f}",
            'l2': f"{l2:.4f}",
            'k3': f"{k3:.4f}",
            'l3': f"{l3:.4f}",
            'k4': f"{k4:.4f}",
            'l4': f"{l4:.4f}"
        })

        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        dy += (l1 + 2 * l2 + 2 * l3 + l4) / 6
        t += h

    return results

@ode_solver_bp.route('/runge-kutta', methods=['GET', 'POST'])
def runge_kutta():
    # Default values
    h = 0.001
    max_t = 0.5
    first_10 = 10
    first_12 = 12
    results = None

    if request.method == 'POST':
        h = float(request.form.get('h', h))
        max_t = float(request.form.get('max_t', max_t))
        first_10 = float(request.form.get('first_10', first_10))
        first_12 = float(request.form.get('first_12', first_12))
        
        results = solve_ode(max_t, h)

    return render_template('runge-kutta.html', 
                           results=results, 
                           h=h, 
                           max_t=max_t, 
                           first_10=first_10, 
                           first_12=first_12)
