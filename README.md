# Create with Conan

```
conan create .
```

# Run locally with Xcode

```
mkdir build && cd build
conan install .. -s build_type=Release -pr:b=default -pr:h=default --build=missing
conan install .. -s build_type=Debug -pr:b=default -pr:h=default --build=missing
cmake .. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=/Users/19293778/src/tensorflow-cpp-conan-example/build/Release/generators/conan_toolchain.cmake -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=Release
```

open tflite-example.xcodeproj project
