import random
from dataclasses import dataclass
from typing import List, Optional
from flask import Flask, Blueprint, render_template, request

# Constants and configuration
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
    state: str = "Waiting"

    @classmethod
    def generate(cls, truck_id):
        rnd_load = random.random()
        truck_load = 10 if rnd_load < 0.5 else 12
        unloading_time = 0.4140 if truck_load == 10 else 0.4300
        return cls(truck_id, truck_load, unloading_time, rnd_load)


class Silo:
    def __init__(self):
        self.flour = 0
        self.state = "Free"

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
            self.state = "Free"
        elif self.flour == SILO_CAPACITY:
            self.state = "Full"
        else:
            self.state = (
                "Supplying Plant" if self.state == "Supplying Plant" else "Free"
            )


class UnloadingArea:
    def __init__(self):
        self.state = "Free"
        self.queue = []
        self.current_silo = None
        self.remaining_truck_load = 0
        self.current_truck = None

    def start_unloading(self, truck: Truck, silo: Silo):
        self.state = "Busy"
        self.current_silo = silo
        self.remaining_truck_load = truck.size
        silo.state = "Being filled"
        self.current_truck = truck
        truck.state = "Unloading"
        return truck.unloading_time

    def finish_unloading(self, silos: List[Silo]):
        if self.current_silo:
            amount_filled = self.current_silo.fill(self.remaining_truck_load)
            self.remaining_truck_load -= amount_filled
            if self.remaining_truck_load > 0:
                new_silo = next((s for s in silos if s.state == "Free"), None)
                if new_silo:
                    self.current_silo = new_silo
                    return SILO_CHANGE_TIME
            self.current_truck.state = "-"
            self.current_truck = None
            self.current_silo = None

        if self.queue:
            next_truck = self.queue.pop(0)
            new_silo = next((s for s in silos if s.state == "Free"), None)
            if new_silo:
                return self.start_unloading(next_truck, new_silo)

        self.state = "Free"
        return 0


class Simulation:
    def __init__(self):
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
        if self.unloading_area.state == "Busy":
            occupation_duration = current_time - self.last_event_time
            self.total_tube_occupation_time += occupation_duration
        self.last_event_time = current_time

    def run_up_to(self, end_row: int) -> None:
        start_row = (
            self.last_simulated_row + 1
        )  # Use last_simulated_row instead of len(self.simulation_data)
        for i in range(start_row, end_row):
            if i == 0:
                self.initialize()
                continue

            event = self.get_next_event()
            self.update_tube_occupation(event["time"])
            self.clock = event["time"]

            if event["type"] == "Truck Arrival":
                self.handle_truck_arrival(event)
            elif event["type"] == "End of Unloading":
                self.handle_end_of_unloading()
            elif event["type"] == "Silo Emptying":
                self.handle_silo_emptying()

            self.update_silo_states()
            self.simulation_data.append(self.create_row(i, event))
            self.last_simulated_row = i  # Update last_simulated_row

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

            if event["type"] == "Truck Arrival":
                self.handle_truck_arrival(event, simulation_data)
            elif event["type"] == "End of Unloading":
                self.handle_end_of_unloading(simulation_data)
            elif event["type"] == "Silo Emptying":
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
        self.simulation_data.append(self.create_row(0, {"type": "Initialization"}))

    def get_next_event(self):
        events = [
            {"type": "Truck Arrival", "time": self.next_arrival},
            {"type": "End of Unloading", "time": self.end_unloading},
            {"type": "Silo Emptying", "time": self.silo_emptying},
        ]
        return min(events, key=lambda x: x["time"])

    def handle_truck_arrival(self, event):
        rnd = random.random()
        time_between_arrivals = (
            ARRIVAL_TIME_RANGE[0]
            + (ARRIVAL_TIME_RANGE[1] - ARRIVAL_TIME_RANGE[0]) * rnd
        )
        self.next_arrival = self.clock + time_between_arrivals

        truck = Truck.generate(self.truck_id_counter)
        self.trucks[self.truck_id_counter] = truck
        self.truck_id_counter += 1

        if self.unloading_area.state == "Free":
            empty_silo = next((s for s in self.silos if s.state == "Free"), None)
            if empty_silo:
                unloading_time = self.unloading_area.start_unloading(truck, empty_silo)
                self.end_unloading = self.clock + unloading_time
            else:
                self.unloading_area.queue.append(truck)
        else:
            self.unloading_area.queue.append(truck)
        self.max_queue = max(self.max_queue, len(self.unloading_area.queue))

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
            self.last_event_type = "Change of Silo"
        else:
            self.end_unloading = float("inf")
            self.total_trucks += 1
            self.last_event_type = "End of Unloading"

        # Check if there's no silo currently supplying the plant
        if self.current_supplying_silo is None:
            # Find the first non-empty silo that's not being filled
            for i, silo in enumerate(self.silos):
                if silo.flour > 0 and silo.state != "Being filled":
                    silo.state = "Supplying Plant"
                    self.current_supplying_silo = i
                    self.silo_emptying = self.clock + 1  # Set next emptying time when starting to supply
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
                    if s.flour > 0 and s.state != "Being filled"
                ),
                None,
            )
            if self.current_supplying_silo is not None:
                self.silos[self.current_supplying_silo].state = "Supplying Plant"
                self.silo_emptying = self.clock + 1  # Set next emptying time when starting to supply

        if self.current_supplying_silo is not None:
            silo = self.silos[self.current_supplying_silo]
            silo.empty(PLANT_CONSUMPTION_RATE)
            if silo.flour == 0:
                silo.state = "Free"
                self.current_supplying_silo = None
                # Find the next silo to supply
                for i, s in enumerate(self.silos):
                    if s.flour > 0 and s.state != "Being filled":
                        s.state = "Supplying Plant"
                        self.current_supplying_silo = i
                        break

        if self.current_supplying_silo is not None:
            self.silo_emptying = self.clock + 1  # Always set next emptying time if a silo is supplying
        else:
            self.silo_emptying = float("inf")

    def update_silo_states(self):
        for i, silo in enumerate(self.silos):
            if i != self.current_supplying_silo:
                if silo.flour > 0 and silo.state not in [
                    "Being filled",
                    "Full",
                    "Supplying Plant",
                ]:
                    silo.update_state()

    def create_row(self, index, event):
        truck_states = {
            truck_id: truck.state for truck_id, truck in self.trucks.items()
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
        if self.last_event_type == "Change of Silo":
            event_type = "Change of Silo"
        else:
            event_type = event["type"]

        # Calculate silo_change_time
        silo_change_time = round(SILO_CHANGE_TIME, 2) if event_type == "Change of Silo" else ""

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
                round(self.unloading_area.remaining_truck_load, 2)
                if isinstance(self.unloading_area.remaining_truck_load, (int, float))
                else ""
            ) if 0 < self.unloading_area.remaining_truck_load < 10 else "",
            **{
                f"truck_{truck_id}_state": state
                for truck_id, state in truck_states.items()
            },
            "tube_queue": len(self.unloading_area.queue),
            "tube_occupation_time": occupation_time,
            "tube_occupation_percentage": occupation_percentage,
            "total_trucks": self.total_trucks,
            "max_queue": self.max_queue,
            "silo_change_time": silo_change_time,

        }


global_simulation = Simulation()

app = Flask(__name__)
final = Blueprint("final", __name__, template_folder="templates")


@final.route("/tp-final", methods=["GET"])
def tp_final():
    simulation_data = None
    max_trucks = 0

    if request.args:
        start_row = int(request.args.get("start_row", 0))
        additional_rows = int(request.args.get("additional_rows", 100))
        total_rows = int(request.args.get("total_rows", 100))

        # Ensure we've simulated up to the required point
        global_simulation.run_up_to(total_rows)

        # Get the first row
        first_row = global_simulation.get_data(0, 1, total_rows)

        # Get the requested range of rows
        middle_rows = global_simulation.get_data(start_row, additional_rows, total_rows)

        # Get the last row
        last_row = global_simulation.get_data(total_rows - 1, 1, total_rows)

        # Combine all rows
        simulation_data = first_row + middle_rows + last_row

        # Remove duplicate rows if any
        simulation_data = [dict(t) for t in {tuple(d.items()) for d in simulation_data}]

        # Sort the rows by index
        simulation_data.sort(key=lambda x: x["index"])

        max_trucks = max(
            len([k for k in row.keys() if k.startswith("truck_") and k.endswith("_state")])
            for row in simulation_data
        )

    return render_template(
        "tp-final.html", simulation_data=simulation_data, max_trucks=max_trucks
    )


if __name__ == "__main__":
    app.register_blueprint(final)
    app.run(debug=True)
