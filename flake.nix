{
    description = "swaptube dev env";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
        microtex-src = {
            url = "github:NanoMichael/MicroTeX";
            flake = false;
        };
    };

    outputs = { self, nixpkgs, flake-utils, microtex-src }:
        flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                    config = { allowUnfree = true; };
                };

                microtex = pkgs.stdenv.mkDerivation {
                    name = "microtex";
                    src = microtex-src;
                    nativeBuildInputs = [
                        pkgs.cmake
                        pkgs.pkg-config
                    ];

                    buildInputs = [
                        pkgs.tinyxml-2
                        pkgs.fontconfig
                        pkgs.gtkmm3
                        pkgs.gtksourceview
                        pkgs.gtksourceviewmm
                    ];

                    installPhase = ''
                        mkdir -p $out/build
                        cp LaTeX $out/build/
                        cp libLaTeX.so $out/build/ 2>/dev/null || true
                        cp -r $src/src $out/
                        cp -r $src/res $out/
                        cp $src/CMakeLists.txt $out/
                    '';

                    noAuditTmpdir = true;
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
                    microtex
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
                    echo "[type ${gpuType}]"
                    export NIX_CFLAGS_COMPILE="-I${pkgs.gdk-pixbuf.dev}/include/gdk-pixbuf-2.0 -I${pkgs.cairo.dev}/include/cairo $NIX_CFLAGS_COMPILE";
                    if [ ! -L "../MicroTeX-master" ] && [ ! -d "../MicroTeX-master" ]; then
                        ln -s ${microtex} ../MicroTeX-master
                    fi

                    trap 'echo "cleaning up symlink"; rm -f ../MicroTeX-master' EXIT # shouldn't delete a cloned repo if it exists for some reason
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
