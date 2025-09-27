import time
from strategy import construir_grid, recomendar_spacing, recomendar_rango
from config import REBALANCE_SECONDS, ORDER_USDT_SIZE, LEVERAGE, SAFE_SPREAD

class GridManager:
    def __init__(self, orders, state):
        self.orders = orders
        self.state = state
        self.grid_active = False
        self.last_signal = None
        self.last_grid_time = 0
        self.current_levels = []
        self.expiry_time = None

    def activate_grid(self, price, signal, expiry=None):
        """Activa el grid tras recibir una señal relevante."""
        self.last_signal = signal
        self.grid_active = True
        self.expiry_time = expiry if expiry is not None else (time.time() + REBALANCE_SECONDS)
        spacing = recomendar_spacing(signal, self.orders.client.config.MIN_GRID_SPACING, self.orders.client.config.MAX_GRID_SPACING)
        rango = recomendar_rango(signal, self.orders.client.config.GRID_RANGE_MIN, self.orders.client.config.GRID_RANGE_MAX)
        self.current_levels = construir_grid(price, spacing, rango)
        self.last_grid_time = time.time()
        self._place_grid_orders()

    def deactivate_grid(self):
        """Desactiva el grid y cancela órdenes abiertas."""
        self.grid_active = False
        self.current_levels = []
        self.orders.cancel_all()
        self.expiry_time = None

    def _place_grid_orders(self):
        """Coloca órdenes limit en los niveles calculados."""
        avg_entry = self.state.calcular_costo_promedio()
        for i, p in enumerate(self.current_levels):
            if p is None or p == 0:
                continue
            if avg_entry and p >= avg_entry:
                continue
            if avg_entry and ((avg_entry - p)/avg_entry < SAFE_SPREAD):
                continue
            qty = self.orders.calcular_cantidad(p, ORDER_USDT_SIZE, LEVERAGE)
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

    def update_on_new_signal(self, price, signal):
        """Actualiza el grid si llega una nueva señal relevante."""
        # Opcional: Puedes agregar lógica para reactivar el grid si la señal cambia
        self.activate_grid(price, signal)

    def status(self):
        return {
            "active": self.grid_active,
            "levels": self.current_levels,
            "expiry": self.expiry_time,
            "last_signal": self.last_signal,
        }
