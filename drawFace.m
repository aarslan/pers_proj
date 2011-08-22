function drawFace(results, rects)

ali = [results{:,:}];
veli = reshape(ali, size(results));

%just take the first guy for now

X = veli(1,4).YhatTest;
Y = veli(2,4).YhatTest;
Z = veli(3,4).YhatTest;

rects = rects(1:numel(X));
coor = [rects{:}];
scatter (coor(1:2:numel(coor)), coor(2:2:numel(coor)), 20, ([X Y (Z+0.5)/2]), 'filled'  )
end