function compilemex( )

try 
    cd bin

    fprintf(1,'Compiling mexsvmlearn\n');
    mex -O  -DMATLAB_MEX -I../src ../src/mexsvmlearn.c ../src/global.c ../src/svm_learn.c ../src/svm_common.c ../src/svm_hideo.c ../src/mexcommon.c 

    fprintf(1,'Compiling mexsvmclassify\n');
    mex -O  -DMATLAB_MEX -I../src  ../src/mexsvmclassify.c ../src/global.c ../src/svm_learn.c ../src/svm_common.c ../src/svm_hideo.c ../src/mexcommon.c 

    fprintf(1,'Compiling mexsinglekernel\n');
    mex -O  -DMATLAB_MEX -I../src ../src/mexsinglekernel.c ../src/global.c ../src/svm_learn.c ../src/svm_common.c ../src/svm_hideo.c ../src/mexcommon.c 
 
    fprintf(1,'Compiling mexkernel\n');
    mex -O  -DMATLAB_MEX -I../src ../src/mexkernel.c ../src/global.c ../src/svm_learn.c ../src/svm_common.c ../src/svm_hideo.c ../src/mexcommon.c 
    
    cd ..
catch
    cd ..
    fprintf(1,'compile failed\n');
end
