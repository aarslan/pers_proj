
results = cell(3,8);
resultsPerm = cell(3,8);
for or=1:3
    or
   parfor cv=1:8
       cv
       [results{or,cv} resultsPerm{or,cv}] = doRegression(origPats, s1vec, truth, or);
   end
end
