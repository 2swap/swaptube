{
    description = "swaptube dev env";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                    config = { allowUnfree = true; };
                };

                basePackages = [
                    pkgs.bashInteractive
                    pkgs.cmake
                    pkgs.ninja
                    pkgs.ffmpeg
                    pkgs.librsvg
                    pkgs.glib
                    pkgs.cairo
                    pkgs.libpng
                    pkgs.gnuplot
                ];

                welcomeHook = gpuType: ''
                    echo "you need microtex to run swaptube, so run:"
                    echo " git clone https://github.com/NanoMichael/MicroTeX.git ../MicroTeX-master"
                    echo "and follow the build instructions at https://github.com/NanoMichael/MicroTeX/"
                    echo "[type ${gpuType}]"
                '';
            in {
                devShells = {
                    default = pkgs.mkShell {
                        buildInputs = basePackages ++ [
                            pkgs.cudatoolkit
                            pkgs.linuxPackages.nvidia_x11
                        ];

                        shellHook = welcomeHook "CUDA";
                    };

                    hip = pkgs.mkShell {
                        buildInputs = basePackages ++ [
                            pkgs.rocmPackages.clr
                        ];

                        shellHook = welcomeHook "HIP/ROCm";
                    };
                };
            });
}
