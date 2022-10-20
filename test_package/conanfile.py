import os
from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.layout import basic_layout


class tflite_exampleTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "VirtualRunEnv"
    apply_env = False
    test_type = "explicit"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        basic_layout(self)

    def test(self):
        if can_run(self):
            model_path = os.path.join(self.source_folder, "../assets", "mobilenet_v1_1.0_224_quant.tflite")
            labels_path = os.path.join(self.source_folder, "../assets", "labels_mobilenet_quant_v1_224.txt")
            image_path = os.path.join(self.source_folder, "../assets", "bird.png")
            self.run(f"tflite-example {model_path} {labels_path} {image_path}", env="conanrun")
