{
    description = "swaptube dev env";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs { inherit system; };
            in {
                devShells.default = pkgs.mkShell {
                    buildInputs = [
                        pkgs.bashInteractive
                        pkgs.cmake
                        pkgs.ninja
                        pkgs.ffmpeg_5
                        pkgs.librsvg
                        pkgs.glib
                        pkgs.cairo
                        pkgs.libpng
                        pkgs.gnuplot
                    ];

                    shellHook = ''
                        echo "you need microtex to run swaptube, so run:"
                        echo " git clone https://github.com/NanoMichael/MicroTeX.git ../MicroTeX-master"
                        echo "and follow the build instructions at https://github.com/NanoMichael/MicroTeX/"
                    '';
                };
            });
}
