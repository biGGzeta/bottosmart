import time
from strategy import construir_grid, recomendar_spacing, recomendar_rango
from config import MIN_GRID_SPACING, MAX_GRID_SPACING, GRID_RANGE_MIN, GRID_RANGE_MAX, REBALANCE_SECONDS, ORDER_USDT_SIZE, LEVERAGE, SAFE_SPREAD

class GridManager:
    def __init__(self, orders, state):
        self.orders = orders
        self.state = state
        # Defaults from config
        self.min_grid_spacing = MIN_GRID_SPACING
        self.max_grid_spacing = MAX_GRID_SPACING
        self.grid_range_min = GRID_RANGE_MIN
        self.grid_range_max = GRID_RANGE_MAX
        self.order_usdt_size = ORDER_USDT_SIZE
        self.leverage = LEVERAGE
        self.safe_spread = SAFE_SPREAD

        # Dynamic state
        self.grid_active = False
        self.last_signal = None
        self.last_grid_time = 0
        self.current_levels = []
        self.expiry_time = None
        self.override_params = {}

    def handle_signal(self, signal, price):
        """Interpret and execute management plan based on signal dict."""
        # Handle dynamic recalibration and TP/SL updates
        recalibrate_grid = signal.get("recalibrate_grid", False)
        update_tp_sl = signal.get("update_tp_sl", False)
        force_regrid = signal.get("force_regrid", False)
        
        # If grid is active and not expired, handle recalibration/updates
        if self.grid_active and not force_regrid:
            if recalibrate_grid:
                self._recalibrate_grid(price, signal)
                self._log_status("Grid recalibrated.")
            
            if update_tp_sl or signal.get("adjust_tp", False) or signal.get("adjust_sl", False):
                self._update_tp_sl(signal)
                self._log_status("TP/SL updated.")
            
            # If we're only recalibrating or updating TP/SL, return early
            if (recalibrate_grid or update_tp_sl) and not signal.get("nuevo_grid", False):
                return

        # Traditional signal handling for new grids or forced regrids
        # Cancel/correct limit orders if instructed
        if signal.get("cancel_all_limits", True):
            self.orders.cancel_all()

        # TP/SL adjust (traditional behavior)
        if signal.get("adjust_tp", True):
            avg = self.state.calcular_costo_promedio()
            pos = float(self.state.state.get('posicion_total', 0.0))
            open_orders = self.orders.get_open_orders()
            self.orders.ensure_take_profits(avg, pos, open_orders, offset=0.0002)
        if signal.get("adjust_sl", True):
            avg = self.state.calcular_costo_promedio()
            sl_price = avg * (1 - self.safe_spread)
            self.orders.colocar_stop_loss_close_position(sl_price)

        # Grid logic - only create new grid if not active, expired, or forced
        if signal.get("nuevo_grid", True):
            if not self.grid_active or force_regrid or (self.expiry_time and time.time() > self.expiry_time):
                self.activate_grid(price, signal)
            else:
                self._log_status("Grid already active and not expired. Skipping new grid creation.")

        # Log plan of action
        self._log_status("Signal handled.")

    def activate_grid(self, price, signal, expiry=None, override=None):
        """Activate grid with optional overrides from signal."""
        self.last_signal = signal
        self.grid_active = True
        self.expiry_time = expiry if expiry is not None else (time.time() + REBALANCE_SECONDS)
        # Use override from signal if present
        params = {
            "min_grid_spacing": self.min_grid_spacing,
            "max_grid_spacing": self.max_grid_spacing,
            "grid_range_min": self.grid_range_min,
            "grid_range_max": self.grid_range_max,
        }
        if override:
            params.update(override)
            self.override_params = override
        elif "override" in signal:
            params.update(signal["override"])
            self.override_params = signal["override"]
        else:
            self.override_params = {}

        spacing = recomendar_spacing(signal, params["min_grid_spacing"], params["max_grid_spacing"])
        rango = recomendar_rango(signal, params["grid_range_min"], params["grid_range_max"])
        self.current_levels = construir_grid(price, spacing, rango)
        self.last_grid_time = time.time()
        self._place_grid_orders()
        self._log_status("Grid activado.")

    def deactivate_grid(self):
        """Deactivate grid and cancel all open orders."""
        self.grid_active = False
        self.current_levels = []
        self.orders.cancel_all()
        self.expiry_time = None
        self.override_params = {}
        self._log_status("Grid desactivado.")

    def _place_grid_orders(self):
        """Place limit orders at grid levels."""
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

    def _recalibrate_grid(self, price, signal):
        """Recalibrate active grid without destroying it completely."""
        if not self.grid_active:
            return
            
        # Use signal overrides if present, otherwise keep current params
        params = {
            "min_grid_spacing": self.min_grid_spacing,
            "max_grid_spacing": self.max_grid_spacing,
            "grid_range_min": self.grid_range_min,
            "grid_range_max": self.grid_range_max,
        }
        
        # Update params from signal override
        if "override" in signal:
            params.update(signal["override"])
            self.override_params.update(signal["override"])
        
        # Calculate new grid levels
        spacing = recomendar_spacing(signal, params["min_grid_spacing"], params["max_grid_spacing"])
        rango = recomendar_rango(signal, params["grid_range_min"], params["grid_range_max"])
        new_levels = construir_grid(price, spacing, rango)
        
        # Use reconcile_grid to update orders efficiently
        avg_entry = self.state.calcular_costo_promedio()
        valid_levels = []
        for p in new_levels:
            if p is None or p == 0:
                continue
            if avg_entry and p >= avg_entry:
                continue
            if avg_entry and ((avg_entry - p)/avg_entry < self.safe_spread):
                continue
            valid_levels.append(p)
        
        if valid_levels:
            qty = self.orders.calcular_cantidad(price, self.order_usdt_size, self.leverage)
            if qty and qty > 0:
                result = self.orders.reconcile_grid(valid_levels, qty)
                print(f"[GRID] Recalibrated: {result}")
        
        # Update current levels
        self.current_levels = new_levels
        self.last_signal = signal

    def _update_tp_sl(self, signal):
        """Update TP/SL independently of grid operations."""
        if signal.get("adjust_tp", True) or signal.get("update_tp_sl", False):
            avg = self.state.calcular_costo_promedio()
            pos = float(self.state.state.get('posicion_total', 0.0))
            open_orders = self.orders.get_open_orders()
            if avg and pos > 0:
                self.orders.ensure_take_profits(avg, pos, open_orders, offset=0.0002)
        
        if signal.get("adjust_sl", True) or signal.get("update_tp_sl", False):
            avg = self.state.calcular_costo_promedio()
            if avg:
                sl_price = avg * (1 - self.safe_spread)
                self.orders.colocar_stop_loss_close_position(sl_price)

    def check_expiry(self):
        """Check if grid should expire and be deactivated."""
        if self.grid_active and self.expiry_time is not None and time.time() > self.expiry_time:
            print("[GRID] Grid expiró, desactivando...")
            self.deactivate_grid()

    def is_active(self):
        return self.grid_active

    def override_config(self, **kwargs):
        """Allow override of parameters in real time."""
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
