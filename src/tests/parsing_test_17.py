# open() is not supported in Online Python Tutor,
# so use io.StringIO to simulate a file (in Python 3)
import io

# create a multi-line string and pass it into StringIO
Code = io.StringIO('''Cnmzkc Ingm Sqtlo hr sgd 45sg zmc btqqdms Oqdrhcdms ne sgd Tmhsdc Rszsdr, hm neehbd rhmbd Izmtzqx 20, 2017.
Adenqd dmsdqhmf onkhshbr, gd vzr z atrhmdrrlzm zmc sdkduhrhnm odqrnmzkhsx.'''
)

# now work with f as though it were an opened file
for line in Code:
        for word in line.strip():
            for letter in word:
                if letter.isalpha:
                    N=ord("letter")
                    O=chr(N+1)
                    out_file.write(O)
                elif letter.isdigit:
                    out_file.write(letter)

