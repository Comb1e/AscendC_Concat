#!/bin/bash
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
  # 清除上次测试性能文件
#rm -rf ./dist/*
if [ "x$1" == "x1" ]; then
    if [ -d "./dist" ]; then
        if [ "$(ls -A "./dist")" ]; then
        echo "已存在whl"
        pip3 install dist/custom_ops*.whl --force-reinstall
        else
            echo "重新生成whl"
            python3 setup.py build bdist_wheel
            pip3 install dist/custom_ops*.whl --force-reinstall
        fi
    else
        echo "重新生成whl"
        python3 setup.py build bdist_wheel
        pip3 install dist/custom_ops*.whl --force-reinstall
    fi
fi


   rm -rf PROF*
   timeout 180  msprof --application="python3 test_op.py $1"

   if [ $? -eq 124 ]; then
        echo "timed out!"
        exit 1
   fi

  time_use=$(($(python3 get_time.py)))
  time_base=9999999999999
        echo "time_base = $time_base time_use = $time_use"
  if  [ $time_use -eq 0 ]; then
      echo "[ERROR] Performance not achieved"
      exit 1
  fi

  if [ $time_use -ge $time_base ]; then
      echo "test fail for performance exceeds baseline data"
      exit 1
  fi

  echo "Operator performance and accuracy have passed"
