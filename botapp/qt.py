import math 

a = int(input("what is value a"))
b = int(input("what is value b"))
c = int(input("what is value c "))

# to find the determinate 
determinate = b**2 - 4*a*c

if determinate >= 0:
    # calcualte the root value 

    root1 = -b + math.sqrt(determinate)/(2*a)
    root2 = -b - math.sqrt(determinate)/(2*a)

    print(f"the root of the quadratic equation id {root1} and {root2}")

else:
    print("yh you got it wrong they not real root")

