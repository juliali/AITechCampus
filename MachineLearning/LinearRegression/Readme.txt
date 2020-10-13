多元梯度下降

【Test 1】
CMD:
-----
$ python gradient_descent_multivariant.py --input input\housing.csv --output output\gradient_descent_multivariant.csv --ignoredColumns ZN,CHAS,NOX,RM,DIS,RAD,TAX,PIRATIO,B,LSTAT --preprocessType normalize --trainingDataRate 0.9 --thresholdRate 0.001 --stepLength 2.0 --maxLoopNumber 50000 --dynamicStep True

Output:
-------
... ...
initial: theta = [ 70610.02035444  -1629.82563376  28027.24077424 -73389.51337578]

[loop 1]: loss = 3977922555.9023476
... ...
[loop 50000]: loss = 69.59816648964065

rmse ratio (rmse / y_mean) is: 0.2918038324169794

execution duration: 0:29:41.696877