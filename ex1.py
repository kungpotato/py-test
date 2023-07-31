
from logger.log import log_debug

# Variables and Data Types
stock = 'META'
price = 325.48
is_expensive = True

log_debug(stock)
log_debug(f'{stock} : {price}')

# Functions


def get_meta_price():
    return price


meta_price = get_meta_price()
log_debug(meta_price)

# Lists, Tuples, and Dictionaries
stocks = ['META', 'TSLA', 'INTC', 'AMD']  # Mutability
stocks[0] = 'APPL'
log_debug(stocks)
my_tuple = (10, 20, 30)  # Immutable
# log_debug(my_tuple)
stock_dict = {'META': 100, 'TSLA': 200, 'INTC': 300, 'AMD': 400}

# Control Flow (if-else, loops)
for st in stocks:
    if st == 'META':
        log_debug(f'{st} : {stock_dict[st]}')
    elif st == 'TSLA':
        log_debug(f'Name : {st}')
    else:
        log_debug(st)

# File Handling
with open("example.txt", "w") as file:
    file.write("This is an example file.\n")

with open("example.txt", "r") as file:
    content = file.read()
    print("File Content:")
    print(content)

# Exception Handling
try:
    result = 10 / 0
except Exception:
    print("Error: Cannot divide by zero.")
else:
    print("Result:", result)
