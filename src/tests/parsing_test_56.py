# open() is not supported in Online Python Tutor,
# so use io.StringIO to simulate a file (in Python 3)
import io

# create a multi-line string and pass it into StringIO
my_file = io.StringIO('''25-03-2016	Popular	$21.30
09-12-2016	NTUC FairPrice	$59.43
08-01-2017	Shell Station	$47.56
09-03-2017	SingTel	$31.50
16-02-2017	Popular	$25.40
08-01-2017	SingTel	$32.10
21-12-2016	SIA	$546.90
14-02-2017	Shaw Theatres	$24.30
19-03-2017	NTUC FairPrice	$108.32

''')

# now work with f as though it were an opened file
mth_list = ['03-2016', '12-2016', '01-2017', '03-2017', '02-2017']

for lines in my_file:
    lines = lines.rstrip("\n")
    columns = lines.split("\t")
    new_date = columns[0]
    price = columns[2]
    price = price.lstrip("$")
    price = float(price)
    total_price = 0
    for date in mth_list:
            if date in new_date:
                total_price += price
                print (date + ": total transaction amount is $" + str(total_price))
