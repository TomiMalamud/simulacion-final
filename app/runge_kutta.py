import matplotlib
matplotlib.use('Agg')

from flask import Blueprint, render_template, request, send_file
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

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
            'yPrime2': y_prime2,
            't': t,
            'y': y,
            'z': dy,
            'k1': k1,
            'l1': l1,
            'k2': k2,
            'l2': l2,
            'k3': k3,
            'l3': l3,
            'k4': k4,
            'l4': l4
        })

        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        dy += (l1 + 2 * l2 + 2 * l3 + l4) / 6
        t += h

        if t > 1:  
            break

    return results

def create_plot(results):
    t_values = [result['t'] for result in results]
    y_prime2_values = [result['yPrime2'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_prime2_values, label="y''(t)")
    plt.xlabel('Time (t)')
    plt.ylabel("y''(t)")
    plt.title("ODE Solution: y''(t) over time")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

@ode_solver_bp.route('/runge-kutta', methods=['GET', 'POST'])
def runge_kutta():
    h = 0.001
    max_t = 1.0  
    first_10 = 10
    first_12 = 12
    results = None
    plot_url = None

    if request.method == 'POST':
        h = float(request.form.get('h', h))
        max_t = float(request.form.get('max_t', max_t))
        first_10 = float(request.form.get('first_10', first_10))
        first_12 = float(request.form.get('first_12', first_12))
        
        results = solve_ode(max_t, h)
        plot_url = create_plot(results)

    return render_template('runge-kutta.html', 
                           results=results, 
                           h=h, 
                           max_t=max_t, 
                           first_10=first_10, 
                           first_12=first_12,
                           plot_url=plot_url)