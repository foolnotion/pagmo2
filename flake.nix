{
  description = "pagmo2 dev";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.url = "github:foolnotion/nur-pkg";

    # make everything follow nixpkgs
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, flake-utils, nixpkgs, foolnotion }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };

        stdenv_ = pkgs.llvmPackages_19.stdenv;

        pagmo2 = stdenv_.mkDerivation {
          name = "pagmo2";
          src = self;

          cmakeFlags = [
            "-DBUILD_SHARED_LIBS=${if pkgs.stdenv.hostPlatform.isStatic then "OFF" else "ON"}"
            "-DPAGMO_BUILD_TESTS=1"
            "-DPAGMO_BUILD_TUTORIALS=OFF"
            "-DPAGMO_WITH_EIGEN3=1"
            "-DPAGMO_WITH_NLOPT=1"
            "-DPAGMO_WITH_IPOPT=1"
          ];

          nativeBuildInputs = with pkgs; [ cmake git ];

          buildInputs = (with pkgs; [
            boost
            cpp-sort
            eigen
            fast-float
            eve
            ipopt
            nlopt
            scnlib
            tbb
          ]);
        };

      in rec {
        packages = {
          default = pagmo2.overrideAttrs(old: {
            cmakeFlags = old.cmakeFlags ++ [
              "-DCMAKE_CXX_FLAGS=${
                if pkgs.stdenv.hostPlatform.isx86_64 then "-march=x86-64-v3" else ""
              }"
            ];
          });
        };

        devShells.default = stdenv_.mkDerivation {
          name = "pagmo";

          nativeBuildInputs = pagmo2.nativeBuildInputs ++ (with pkgs; [
            clang-tools_19
            cppcheck
            include-what-you-use
            cmake-language-server
          ]);

          buildInputs = pagmo2.buildInputs ++ (with pkgs; [
            gdb
            gcc13
            graphviz
            hyperfine
            linuxPackages_latest.perf
            seer
            valgrind
            hotspot
            mold
          ]);

          shellHook = ''
            export LD_LIBRARY_PATH=$CMAKE_LIBRARY_PATH
            alias bb="cmake --build build -j"
          '';
        };
      });
}
