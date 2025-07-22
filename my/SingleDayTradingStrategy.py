class SingleDayTradingStrategy:
    
    def __init__(self, date, cash, window):
        self.date = date
        self.cash = cash
        self.window = window
        self.position = 0
        self.entry = None
        self.trades = []
        self.equity_curve = []
        self.exception_bars = []
    
    def _open_position(self, position, price, now):
        if self.position == -position:
            self._close_position(price, now)
        self.position = position
        self.entry = {'time': now, 'price': price}

    def _close_position(self, price, now):
        if self.position == 0 or self.entry is None:
            return
        
        pnl = (price - self.entry['price']) * self.position
        self.cash += pnl
        self.trades.append({
            'date': self.date,
            'entry_time': self.entry['time'],
            'exit_time': now,
            'entry_price': self.entry['price'],
            'exit_price': price,
            'position': self.position,
            'pnl': pnl
        })
        self.position = 0
        self.entry = None

    def run_single_day(self, day):
        self.position = 0
        self.entry = None
        self.trades = []
        self.equity_curve = []

        day['rolling_high'] = day['high'].shift(1).rolling(self.window).max()
        day['rolling_low'] = day['low'].shift(1).rolling(self.window).min()

        for i in range(self.window, len(day)):
            row = day.iloc[i]
            now = row.name
            rowLow = row['low']
            rowHigh = row['high']
            highN = row['rolling_high']
            lowN = row['rolling_low']

            # Close position at end of day
            if i == len(day) - 1:
                price = row['close']
                self._close_position(price, now)

            elif rowHigh > highN and rowLow < lowN:
                self.exception_bars.append(row)
                continue

            elif rowHigh > highN and self.position <= 0:
                price = highN
                self._open_position(1, price, now)

            elif rowLow < lowN and self.position >= 0:
                price = lowN
                self._open_position(-1, price, now)

            if self.position != 0 and self.entry:
                unrealized = (price - self.entry['price']) * self.position
            else:
                unrealized = 0
            equity = self.cash + unrealized
            self.equity_curve.append({'time': now, 'equity': equity})

        return self.trades, self.equity_curve, self.cash, self.exception_bars