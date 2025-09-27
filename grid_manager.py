import time
from strategy import construir_grid, recomendar_spacing, recomendar_rango
from config import MIN_GRID_SPACING, MAX_GRID_SPACING, GRID_RANGE_MIN, GRID_RANGE_MAX, REBALANCE_SECONDS, ORDER_USDT_SIZE, LEVERAGE, SAFE_SPREAD

class GridManager:
    def __init__(self, orders, state):
        self.orders = orders
        self.state = state
        # Valores base (defaults de config)
        self.min_grid_spacing = MIN_GRID_SPACING
        self.max_grid_spacing = MAX_GRID_SPACING
        self.grid_range_min = GRID_RANGE_MIN
        self.grid_range_max = GRID_RANGE_MAX
        self.order_usdt_size = ORDER_USDT_SIZE
        self.leverage = LEVERAGE
        self.safe_spread = SAFE_SPREAD

        # Estado dinámico
        self.grid_active = False
        self.last_signal = None
        self.last_grid_time = 0
        self.current_levels = []
        self.expiry_time = None
        self.override_params = {}

    def activate_grid(self, price, signal, expiry=None, override=None):
        """Activa el grid tras recibir una señal relevante, con opcional override."""
        self.last_signal = signal
        self.grid_active = True
        self.expiry_time = expiry if expiry is not None else (time.time() + REBALANCE_SECONDS)
        # Override dinámico
        params = {
            "min_grid_spacing": self.min_grid_spacing,
            "max_grid_spacing": self.max_grid_spacing,
            "grid_range_min": self.grid_range_min,
            "grid_range_max": self.grid_range_max,
        }
        if override:
            params.update(override)
            self.override_params = override
        else:
            self.override_params = {}

        spacing = recomendar_spacing(signal, params["min_grid_spacing"], params["max_grid_spacing"])
        rango = recomendar_rango(signal, params["grid_range_min"], params["grid_range_max"])
        self.current_levels = construir_grid(price, spacing, rango)
        self.last_grid_time = time.time()
        self._place_grid_orders()
        self._log_status("Grid activado.")

    def deactivate_grid(self):
        """Desactiva el grid y cancela órdenes abiertas."""
        self.grid_active = False
        self.current_levels = []
        self.orders.cancel_all()
        self.expiry_time = None
        self.override_params = {}
        self._log_status("Grid desactivado.")

    def _place_grid_orders(self):
        """Coloca órdenes limit en los niveles calculados."""
        avg_entry = self.state.calcular_costo_promedio()
        for i, p in enumerate(self.current_levels):
            if p is None or p == 0:
                continue
            if avg_entry and p >= avg_entry:
                continue
            if avg_entry and ((avg_entry - p)/avg_entry < self.safe_spread):
                continue
            qty = self.orders.calcular_cantidad(p, self.order_usdt_size, self.leverage)
            if qty is None or qty == 0:
                continue
            try:
                self.orders.place_grid_buy(p, qty, i)
            except Exception as e:
                print(f"[ERROR] crear orden grid: {e}")

    def check_expiry(self):
        """Chequea si el grid debe expirar y ser desactivado."""
        if self.grid_active and self.expiry_time is not None and time.time() > self.expiry_time:
            print("[GRID] Grid expiró, desactivando...")
            self.deactivate_grid()

    def is_active(self):
        return self.grid_active

    def override_config(self, **kwargs):
        """Permite override de parámetros en tiempo real."""
        self.override_params.update(kwargs)

    def reset_override(self):
        self.override_params = {}

    def status(self):
        return {
            "active": self.grid_active,
            "levels": self.current_levels,
            "expiry": self.expiry_time,
            "last_signal": self.last_signal,
            "override": self.override_params,
        }

    def _log_status(self, msg=""):
        print(f"[GRID_MANAGER] {msg} status={self.status()}")
