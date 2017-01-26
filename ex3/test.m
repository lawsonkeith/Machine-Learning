%n=10
%b = [1 2 3 4 5 6 7 8 9 0]
%for i = 1:n-1
%  a(i) = b(i+1) - b(i);
%endfor
%a
%a = b(2:n) - b(1:n-1);
%a

Q = zeros(5,3)      % create a test matrix of all-zeros
Q
v = [1 2 3]'        % create a column vector
v
Q(2,:) = v          % copy v into the 2nd row of Q
Q
Q'
