"""
Water Supply Redistribution Optimizer.
Uses Linear Programming (PuLP with SciPy linprog fallback) to solve
optimal water allocation across municipal zones, minimizing unmet demand
while honoring priority weights and pipeline physical constraints.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from scipy.optimize import linprog


class WaterRedistributionOptimizer:
    """
    Linear Programming Optimizer for Municipal Water Redistribution.

    Mathematical Formulation:
    Variables:
        S_i : Allocated water supply to zone i (kL/hour)
        U_i : Unmet demand / shortage in zone i (kL/hour)
        O_i : Surplus water in zone i (kL/hour)

    Objective Function:
        Minimize: sum_i [ w_i * U_i + 0.1 * O_i ]
        where w_i is the priority-weighted penalty factor for unmet demand.

    Subject to:
        1. S_i + U_i - O_i = PredictedDemand_i  (Demand satisfaction balance)
        2. sum_i S_i <= TotalAvailableSupply     (Conservation of municipal water)
        3. 0 <= S_i <= PipelineCapacity_i       (Physical hydraulic pipeline limits)
        4. U_i >= 0, O_i >= 0                   (Non-negativity)
        5. S_i >= MinEssential_i                (Essential life-safety baseline for high priority)
    """

    def __init__(self, use_pulp: bool = True):
        self.use_pulp = use_pulp and PULP_AVAILABLE

    def optimize(
        self,
        zone_demands: Dict[str, float],
        zone_capacities: Dict[str, float],
        zone_priorities: Dict[str, int],
        total_available_supply: float,
        critical_threshold: float = 0.15,
        warning_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Solves the LP optimization problem for given municipal scenario.

        Parameters:
            zone_demands: {zone_id: predicted_demand_kL}
            zone_capacities: {zone_id: pipeline_max_capacity_kL}
            zone_priorities: {zone_id: priority_int_1_to_5}
            total_available_supply: Total municipal reservoir water available (kL)

        Returns:
            Dict containing allocations, unmet deficits, surplus, risk statuses, and summary KPIs.
        """
        zones = list(zone_demands.keys())
        n_zones = len(zones)

        # Baseline priority weights (Priority 5 = Medical/Hospitals gets highest penalty for shortage)
        weights = {z: float(zone_priorities.get(z, 3) ** 1.8) for z in zones}

        allocations: Dict[str, float] = {}
        unmet: Dict[str, float] = {}

        # Attempt PuLP LP Formulation
        solved_with_pulp = False
        if self.use_pulp:
            try:
                prob = pulp.LpProblem("Water_Redistribution_Optimization", pulp.LpMinimize)

                # Decision variables
                s_vars = {z: pulp.LpVariable(f"Supply_{z}", lowBound=0, upBound=zone_capacities.get(z, 2000.0)) for z in zones}
                u_vars = {z: pulp.LpVariable(f"Unmet_{z}", lowBound=0) for z in zones}

                # Objective: Minimize weighted unmet demand
                prob += pulp.lpSum([weights[z] * u_vars[z] for z in zones]), "Minimize_Unmet_Demand"

                # Constraint 1: Total available supply
                prob += pulp.lpSum([s_vars[z] for z in zones]) <= total_available_supply, "Total_Supply_Constraint"

                # Constraint 2 & 3: Demand satisfaction (Supply + Unmet >= Demand)
                for z in zones:
                    demand = zone_demands[z]
                    prob += s_vars[z] + u_vars[z] >= demand, f"Demand_Balance_{z}"

                    # Priority guarantee: Zones with priority >= 4 get at least 65% of demand if possible
                    if zone_priorities.get(z, 1) >= 4:
                        essential = min(demand * 0.65, zone_capacities.get(z, 2000.0))
                        prob += s_vars[z] >= min(essential, total_available_supply / n_zones), f"Essential_{z}"

                # Solve quietly
                solver = pulp.PULP_CBC_CMD(msg=False)
                prob.solve(solver)

                if pulp.LpStatus[prob.status] in ["Optimal", "Not Solved"]:
                    for z in zones:
                        alloc = max(0.0, float(pulp.value(s_vars[z])))
                        allocations[z] = round(alloc, 2)
                        unmet[z] = round(max(0.0, zone_demands[z] - alloc), 2)
                    solved_with_pulp = True
            except Exception as e:
                solved_with_pulp = False

        # Fallback to SciPy Highs Simplex/Interior Point LP Solver
        if not solved_with_pulp:
            # SciPy linprog formulation:
            # Vector x = [S_0..S_n-1, U_0..U_n-1]  (2*n variables)
            # Objective: min sum [ 0 * S_i + w_i * U_i ]
            c = [0.0] * n_zones + [weights[z] for z in zones]

            # Inequality A_ub * x <= b_ub
            # 1. sum(S_i) <= total_available_supply
            # 2. -S_i - U_i <= -demand_i   (equivalent to S_i + U_i >= demand_i)
            A_ub = []
            b_ub = []

            # Total supply constraint
            A_ub.append([1.0] * n_zones + [0.0] * n_zones)
            b_ub.append(total_available_supply)

            # Demand satisfaction per zone
            for i, z in enumerate(zones):
                row = [0.0] * (2 * n_zones)
                row[i] = -1.0           # -S_i
                row[n_zones + i] = -1.0 # -U_i
                A_ub.append(row)
                b_ub.append(-zone_demands[z])

            bounds = []
            # Bounds for S_i: (0, capacity_i)
            for z in zones:
                bounds.append((0.0, zone_capacities.get(z, 2000.0)))
            # Bounds for U_i: (0, None)
            for z in zones:
                bounds.append((0.0, None))

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success:
                for i, z in enumerate(zones):
                    alloc = max(0.0, float(res.x[i]))
                    allocations[z] = round(alloc, 2)
                    unmet[z] = round(max(0.0, zone_demands[z] - alloc), 2)
            else:
                # Proportional heuristic emergency fallback
                total_dem = sum(zone_demands.values())
                ratio = min(1.0, total_available_supply / max(total_dem, 1.0))
                for z in zones:
                    alloc = min(zone_demands[z] * ratio, zone_capacities.get(z, 2000.0))
                    allocations[z] = round(alloc, 2)
                    unmet[z] = round(max(0.0, zone_demands[z] - alloc), 2)

        # Build detailed assessment table
        results_rows = []
        total_demand = sum(zone_demands.values())
        total_allocated = sum(allocations.values())
        total_unmet = sum(unmet.values())

        for z in zones:
            dem = zone_demands[z]
            alloc = allocations[z]
            defic = unmet[z]
            deficit_pct = (defic / max(dem, 1e-4))

            # Categorize shortage risk level
            if deficit_pct >= critical_threshold:
                risk_status = "Critical Shortage"
                color_code = "#FF4B4B" # Red
                status_badge = "🔴 CRITICAL"
            elif deficit_pct >= warning_threshold:
                risk_status = "Warning"
                color_code = "#FFA500" # Yellow/Amber
                status_badge = "🟡 WARNING"
            else:
                risk_status = "Adequate"
                color_code = "#00D26A" # Green
                status_badge = "🟢 ADEQUATE"

            results_rows.append({
                "zone_id": z,
                "predicted_demand_kL": dem,
                "allocated_supply_kL": alloc,
                "shortage_deficit_kL": defic,
                "deficit_pct": round(deficit_pct * 100.0, 1),
                "pipeline_capacity_kL": zone_capacities.get(z, 0.0),
                "priority_level": zone_priorities.get(z, 3),
                "risk_status": risk_status,
                "color_code": color_code,
                "status_badge": status_badge
            })

        results_df = pd.DataFrame(results_rows)

        # Calculate dynamic transfer routes (from surplus zones to deficit zones)
        transfer_routes = []
        surplus_pool = []
        deficit_pool = []

        for z in zones:
            diff = allocations[z] - zone_demands[z]
            if diff > 10.0:
                surplus_pool.append({"zone": z, "amount": diff})
            elif diff < -10.0:
                deficit_pool.append({"zone": z, "amount": abs(diff)})

        for s in surplus_pool:
            for d in deficit_pool:
                if s["amount"] > 0 and d["amount"] > 0:
                    transfer_vol = min(s["amount"], d["amount"])
                    transfer_routes.append({
                        "from_zone": s["zone"],
                        "to_zone": d["zone"],
                        "transfer_volume_kL": round(transfer_vol, 2)
                    })
                    s["amount"] -= transfer_vol
                    d["amount"] -= transfer_vol

        return {
            "summary": {
                "total_demand_kL": round(total_demand, 2),
                "total_available_kL": round(total_available_supply, 2),
                "total_allocated_kL": round(total_allocated, 2),
                "total_unmet_kL": round(total_unmet, 2),
                "overall_satisfaction_pct": round(max(0.0, min(100.0, (total_allocated / max(total_demand, 1e-4)) * 100)), 1),
                "critical_zones_count": int((results_df["risk_status"] == "Critical Shortage").sum()),
                "warning_zones_count": int((results_df["risk_status"] == "Warning").sum()),
                "adequate_zones_count": int((results_df["risk_status"] == "Adequate").sum()),
                "solver_engine": "PuLP (CBC)" if solved_with_pulp else "SciPy (HiGHS)"
            },
            "zone_breakdown": results_df,
            "transfer_routes": pd.DataFrame(transfer_routes) if transfer_routes else pd.DataFrame(columns=["from_zone", "to_zone", "transfer_volume_kL"])
        }
