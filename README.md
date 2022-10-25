# Create with Conan

```
conan create .
```

# Run locally with Xcode

```
mkdir build && cd build
conan install .. -s build_type=Release -pr:b=default -pr:h=default --build=missing
conan install .. -s build_type=Debug -pr:b=default -pr:h=default --build=missing
cmake .. -G Xcode -DCMAKE_TOOLCHAIN_FILE=generators/conan_toolchain.cmake
```

open tflite-example.xcodeproj project
