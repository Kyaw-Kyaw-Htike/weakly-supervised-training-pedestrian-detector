function mkdir2(str_dir)
% Copyright (C) Dr. Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
% similar to the built-in mkdir with the difference that 
% if overwrites any existing directory with the given name.
% IOW, mkdir2 always make a new directory (with no contents therein)

if exist(str_dir, 'dir')==7
    rmdir(str_dir, 's'); 
end; 

mkdir(str_dir);

end
