import random
from dataclasses import dataclass
from typing import List
from flask import Flask, Blueprint, render_template, request

# Constantes y configuración
SILO_CAPACITY = 20
NUM_SILOS = 4
ARRIVAL_TIME_RANGE = (5, 9)
SILO_CHANGE_TIME = 1 / 6
PLANT_CONSUMPTION_RATE = 0.5
PLANT_CONSUMPTION_INTERVAL = 1


@dataclass
class Truck:
    id: int
    size: int
    unloading_time: float
    rnd_load: float
    state: str = "Esperando"
    load: int = 0

    @classmethod
    def generate(cls, truck_id):
        rnd_load = random.random()
        truck_load = 10 if rnd_load < 0.5 else 12
        unloading_time = 0.4140 if truck_load == 10 else 0.4300
        return cls(truck_id, truck_load, unloading_time, rnd_load, load=truck_load)  


class Silo:
    def __init__(self):
        self.flour = 0
        self.state = "Libre"

    @property
    def is_full(self):
        return self.flour >= SILO_CAPACITY

    def fill(self, amount):
        space_available = SILO_CAPACITY - self.flour
        amount_filled = min(amount, space_available)
        self.flour += amount_filled
        self.update_state()
        return amount_filled

    def empty(self, amount):
        amount_emptied = min(amount, self.flour)
        self.flour -= amount_emptied
        self.update_state()
        return amount_emptied

    def update_state(self):
        if self.flour == 0:
            self.state = "Libre"
        elif self.flour == SILO_CAPACITY:
            self.state = "Lleno"
        else:
            self.state = (
                "Surtiendo Planta" if self.state == "Surtiendo Planta" else "Libre"
            )


class UnloadingArea:
    def __init__(self):
        self.state = "Libre"
        self.queue = []
        self.current_silo = None
        self.remaining_truck_load = 0
        self.current_truck = None

    def start_unloading(self, truck: Truck, silo: Silo):
        self.state = "Ocupado"
        self.current_silo = silo
        self.remaining_truck_load = truck.load
        silo.state = "Siendo rellenado"
        self.current_truck = truck
        truck.state = "Descargando"
        return truck.unloading_time

    def finish_unloading(self, silos: List[Silo]):
        if self.current_silo:
            amount_filled = self.current_silo.fill(self.remaining_truck_load)
            self.remaining_truck_load -= amount_filled
            self.current_truck.load -= amount_filled
            if self.remaining_truck_load > 0:
                new_silo = next((s for s in silos if s.state == "Libre"), None)
                if new_silo:
                    self.current_silo = new_silo
                    return SILO_CHANGE_TIME
            self.current_truck.state = "-"
            self.current_truck = None
            self.current_silo = None

        self.state = "Libre"
        return 0

class Simulation:
    def __init__(
        self,
        arrival_time_range=(5, 9),
        silo_change_time=round(1 / 6, 2),
        plant_consumption_rate=0.5,
    ):
        self.arrival_time_range = arrival_time_range
        self.silo_change_time = silo_change_time
        self.plant_consumption_rate = plant_consumption_rate
        self.clock = 0
        self.next_arrival = 0
        self.end_unloading = float("inf")
        self.silo_emptying = float("inf")
        self.silos = [Silo() for _ in range(NUM_SILOS)]
        self.unloading_area = UnloadingArea()
        self.current_supplying_silo = None
        self.truck_id_counter = 0
        self.trucks = {}
        self.simulation_data = []
        self.last_simulated_row = -1
        self.tube_occupation_start = 0
        self.total_tube_occupation_time = 0
        self.last_event_time = 0
        self.total_trucks = 0
        self.max_queue = 0
        self.last_event_type = None

    def update_tube_occupation(self, current_time):
        if self.unloading_area.state == "Ocupado":
            occupation_duration = current_time - self.last_event_time
            self.total_tube_occupation_time += occupation_duration
        self.last_event_time = current_time

    def run_up_to(self, end_row: int) -> None:
        start_row = (
            self.last_simulated_row + 1
        ) 
        for i in range(start_row, end_row):
            if i == 0:
                self.initialize()
                continue

            event = self.get_next_event()
            self.update_tube_occupation(event["time"])
            self.clock = event["time"]

            if event["type"] == "Llegada de camión":
                self.handle_truck_arrival(event)
            elif event["type"] == "Fin de descarga":
                self.handle_end_of_unloading()
            elif event["type"] == "Vaciado de Silo":
                self.handle_silo_emptying()

            self.update_silo_states()
            self.simulation_data.append(self.create_row(i, event))
            self.last_simulated_row = i 

    def get_data(
        self, start_row: int, additional_rows: int, total_rows: int
    ) -> List[dict]:
        end_row = min(start_row + additional_rows, total_rows)
        if end_row > self.last_simulated_row + 1:
            self.run_up_to(end_row)
        return self.simulation_data[start_row:end_row]

    def run(self, start_row: int, additional_rows: int) -> List[dict]:
        simulation_data = []
        for i in range(start_row, start_row + additional_rows):
            if i == 0:
                self.initialize(simulation_data)
                continue

            event = self.get_next_event()
            self.clock = event["time"]

            if event["type"] == "Llegada de camión":
                self.handle_truck_arrival(event, simulation_data)
            elif event["type"] == "Fin de descarga":
                self.handle_end_of_unloading(simulation_data)
            elif event["type"] == "Vaciado de Silo":
                self.handle_silo_emptying(simulation_data)

            self.update_silo_states()
            simulation_data.append(self.create_row(i, event))

        return simulation_data

    def initialize(self):
        rnd = random.random()
        time_between_arrivals = (
            ARRIVAL_TIME_RANGE[0]
            + (ARRIVAL_TIME_RANGE[1] - ARRIVAL_TIME_RANGE[0]) * rnd
        )
        self.next_arrival = self.clock + time_between_arrivals
        self.simulation_data.append(self.create_row(0, {"type": "Inicialización"}))

    def get_next_event(self):
        events = [
            {"type": "Llegada de camión", "time": self.next_arrival},
            {"type": "Fin de descarga", "time": self.end_unloading},
            {"type": "Vaciado de Silo", "time": self.silo_emptying},
        ]
        return min(events, key=lambda x: x["time"])

    def handle_truck_arrival(self, event):
        rnd = random.random()
        time_between_arrivals = (
            self.arrival_time_range[0]
            + (self.arrival_time_range[1] - self.arrival_time_range[0]) * rnd
        )
        self.next_arrival = self.clock + time_between_arrivals

        truck = Truck.generate(self.truck_id_counter)
        self.trucks[self.truck_id_counter] = truck
        self.truck_id_counter += 1

        self.unloading_area.queue.append(truck)
        self.max_queue = max(self.max_queue, len(self.unloading_area.queue))

        if self.unloading_area.state == "Libre":
            self.start_next_unloading()

        event.update(
            {
                "rnd": rnd,
                "time_between_arrivals": time_between_arrivals,
                "truck_load": truck.size,
                "unloading_time": truck.unloading_time,
                "rnd_load": truck.rnd_load,
                "truck_id": truck.id,
            }
        )


    def handle_end_of_unloading(self):
        additional_time = self.unloading_area.finish_unloading(self.silos)
        if additional_time > 0:
            self.end_unloading = self.clock + additional_time
            self.last_event_type = "Cambio de Silo"
        else:
            self.end_unloading = float("inf")
            self.total_trucks += 1
            self.last_event_type = "Fin de descarga"
            self.start_next_unloading()

        if self.current_supplying_silo is None:        
            for i, silo in enumerate(self.silos):
                if silo.flour > 0 and silo.state != "Siendo rellenado":
                    silo.state = "Surtiendo Planta"
                    self.current_supplying_silo = i
                    self.silo_emptying = self.clock + 1
                    break

        self.max_queue = max(self.max_queue, len(self.unloading_area.queue))

    def handle_silo_emptying(self):
        if (
            self.current_supplying_silo is None
            or self.silos[self.current_supplying_silo].flour == 0
        ):
            self.current_supplying_silo = next(
                (
                    i
                    for i, s in enumerate(self.silos)
                    if s.flour > 0 and s.state != "Siendo rellenado"
                ),
                None,
            )
            if self.current_supplying_silo is not None:
                self.silos[self.current_supplying_silo].state = "Surtiendo Planta"
                self.silo_emptying = self.clock + 1
            else:
                # All silos are empty or being filled, check if we can start filling an empty silo
                self.check_and_start_filling()

        if self.current_supplying_silo is not None:
            silo = self.silos[self.current_supplying_silo]
            silo.empty(self.plant_consumption_rate)
            if silo.flour == 0:
                silo.state = "Libre"
                self.current_supplying_silo = None
                # Check if we can start filling this newly emptied silo
                self.check_and_start_filling()
                
                # Find next silo to supply the plant
                for i, s in enumerate(self.silos):
                    if s.flour > 0 and s.state != "Siendo rellenado":
                        s.state = "Surtiendo Planta"
                        self.current_supplying_silo = i
                        break

        if self.current_supplying_silo is not None:
            self.silo_emptying = self.clock + 1
        else:
            self.silo_emptying = float("inf")
    def check_and_start_filling(self):
        if self.unloading_area.state == "Libre" and self.unloading_area.queue:
            empty_silo = next((s for s in self.silos if s.state == "Libre"), None)
            if empty_silo:
                next_truck = self.unloading_area.queue.pop(0)
                unloading_time = self.unloading_area.start_unloading(next_truck, empty_silo)
                self.end_unloading = self.clock + unloading_time

    def start_next_unloading(self):
        if self.unloading_area.queue:
            empty_silo = next((s for s in self.silos if s.state == "Libre"), None)
            if empty_silo:
                next_truck = self.unloading_area.queue.pop(0)
                unloading_time = self.unloading_area.start_unloading(next_truck, empty_silo)
                self.end_unloading = self.clock + unloading_time

    def update_silo_states(self):
        for i, silo in enumerate(self.silos):
            if i != self.current_supplying_silo:
                if silo.flour > 0 and silo.state not in [
                    "Siendo rellenado",
                    "Lleno",
                    "Surtiendo Planta",
                ]:
                    silo.update_state()

    def create_row(self, index, event):
        truck_states = {
            truck_id: truck.state for truck_id, truck in self.trucks.items()
        }
        truck_loads = {
        truck_id: truck.load for truck_id, truck in self.trucks.items()
        }

        occupation_time = round(self.total_tube_occupation_time, 2)
        occupation_percentage = round(
            (
                (self.total_tube_occupation_time / self.clock) * 100
                if self.clock > 0
                else 0
            ),
            2,
        )
        if self.last_event_type == "Cambio de Silo":
            event_type = "Cambio de Silo"
        elif self.last_event_type == "Continuación de descarga":
            event_type = "Continuación de descarga"
        else:
            event_type = event["type"]
        
        silo_change_time = (
            round(SILO_CHANGE_TIME, 2) if event_type == "Cambio de Silo" else ""
        )

        return {
            "index": index,
            "event": event_type,
            "clock": round(self.clock, 2),
            "rnd": (
                round(event.get("rnd", ""), 2)
                if isinstance(event.get("rnd"), float)
                else ""
            ),
            "time_between_arrivals": (
                round(event.get("time_between_arrivals", ""), 2)
                if event.get("time_between_arrivals")
                else ""
            ),
            "next_arrival": round(self.next_arrival, 2),
            "rnd_load": (
                round(event.get("rnd_load", ""), 2)
                if isinstance(event.get("rnd_load"), float)
                else ""
            ),
            "truck_load": event.get("truck_load", ""),
            "unloading_time": event.get("unloading_time", ""),
            "end_unloading": (
                round(self.end_unloading, 2)
                if self.end_unloading != float("inf")
                else ""
            ),
            "silo_emptying": (
                round(self.silo_emptying, 2)
                if self.silo_emptying != float("inf")
                else ""
            ),
            **{f"silo_{i+1}_state": silo.state for i, silo in enumerate(self.silos)},
            **{
                f"silo_{i+1}_flour": round(silo.flour, 2)
                for i, silo in enumerate(self.silos)
            },
            "tube_queue": self.unloading_area.queue,
            "tube_state": self.unloading_area.state,
            "remaining_truck_load": (
                (
                    round(self.unloading_area.remaining_truck_load, 2)
                    if isinstance(
                        self.unloading_area.remaining_truck_load, (int, float)
                    )
                    else ""
                )
                if 0 < self.unloading_area.remaining_truck_load < 10
                else ""
            ),
            **{
                f"truck_{truck_id}_state": state
                for truck_id, state in truck_states.items()
            },
            **{
            f"truck_{truck_id}_load": load
            for truck_id, load in truck_loads.items()
           },
            "tube_queue": len(self.unloading_area.queue),
            "tube_occupation_time": occupation_time,
            "tube_occupation_percentage": occupation_percentage,
            "total_trucks": self.total_trucks,
            "max_queue": self.max_queue if (len(self.unloading_area.queue) > 0 and self.max_queue >0) else 0,
            "silo_change_time": silo_change_time,
        }


app = Flask(__name__)
final = Blueprint("final", __name__, template_folder="templates")

global_simulation = None  


@final.route("/tp-final", methods=["GET"])
def tp_final():
    simulation_data = None
    max_trucks = 0

    arrival_time_min = float(request.args.get("arrival_time_min", 5))
    arrival_time_max = float(request.args.get("arrival_time_max", 9))
    silo_change_time = float(request.args.get("silo_change_time", round(1 / 6, 2)))
    plant_consumption_rate = float(request.args.get("plant_consumption_rate", 0.5))

    start_row = int(request.args.get("start_row", 0))
    additional_rows = int(request.args.get("additional_rows", 100))
    total_rows = int(request.args.get("total_rows", 100))

    simulation = Simulation(
        arrival_time_range=(arrival_time_min, arrival_time_max),
        silo_change_time=silo_change_time,
        plant_consumption_rate=plant_consumption_rate,
    )

    if request.args:        

        simulation.run_up_to(total_rows)        
        first_row = simulation.get_data(0, 1, total_rows)        
        middle_rows = simulation.get_data(start_row, additional_rows, total_rows)
        last_row = simulation.get_data(total_rows - 1, 1, total_rows)
        simulation_data = first_row + middle_rows + last_row
        simulation_data = [dict(t) for t in {tuple(d.items()) for d in simulation_data}]
        simulation_data.sort(key=lambda x: x["index"])

        max_trucks = max(
            len(
                [
                    k
                    for k in row.keys()
                    if k.startswith("truck_") and k.endswith("_state")
                ]
            )
            for row in simulation_data
        )

    return render_template(
        "tp-final.html",
        simulation_data=simulation_data,
        max_trucks=max_trucks,
        current_params={
            "arrival_time_min": arrival_time_min,
            "arrival_time_max": arrival_time_max,
            "silo_change_time": silo_change_time,
            "plant_consumption_rate": plant_consumption_rate,
            "start_row": start_row,
            "additional_rows": additional_rows,
            "total_rows": total_rows,
        },
    )


if __name__ == "__main__":
    app.register_blueprint(final)
    app.run(debug=True)
