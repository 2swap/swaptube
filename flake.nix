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
                    pkgs.pkg-config 
                    pkgs.llvmPackages.lld
                    pkgs.glib.dev
                    pkgs.xz
                    pkgs.gdk-pixbuf
                ];

                rocmMerged = pkgs.symlinkJoin {
                    name = "rocm-merged";
                    paths = [
                        pkgs.rocmPackages.clr
                        pkgs.rocmPackages.hipify
                        pkgs.rocmPackages.rocprim
                        pkgs.rocmPackages.rocthrust
                    ];
                };

                welcomeHook = gpuType: ''
                    echo "you need microtex to run swaptube, so run:"
                    echo " git clone https://github.com/NanoMichael/MicroTeX.git ../MicroTeX-master"
                    echo "and follow the build instructions at https://github.com/NanoMichael/MicroTeX/"
                    echo "[type ${gpuType}]"
                    export NIX_CFLAGS_COMPILE="-I${pkgs.gdk-pixbuf.dev}/include/gdk-pixbuf-2.0 -I${pkgs.cairo.dev}/include/cairo $NIX_CFLAGS_COMPILE";
                '';
            in {
                devShells = {
                    default = pkgs.mkShell {
                        buildInputs = basePackages ++ [
                            pkgs.cudatoolkit
                        ];

                        shellHook = ''
                            ${welcomeHook "CUDA"}
                            export CUDA_PATH="${pkgs.cudatoolkit}"
                        '';
                    };

                    hip = pkgs.mkShell {
                        buildInputs = basePackages ++ [
                            rocmMerged
                        ];

                        shellHook = ''
                            ${welcomeHook "HIP/ROCm"}
                            export ROCM_PATH="${rocmMerged}"
                            export HIP_PATH="${rocmMerged}"
                            export NIX_CFLAGS_COMPILE="-I${rocmMerged}/include $NIX_CFLAGS_COMPILE";
                        '';
                    };
                };
            });
}
