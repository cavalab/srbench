{
  description = "SRBench";

  inputs = {
    pypi-deps-db.url = "github:DavHau/pypi-deps-db";
    flake-utils.url = "github:numtide/flake-utils";

    mach-nix = {
      url = "github:DavHau/mach-nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
        pypi-deps-db.follows = "pypi-deps-db";
      };
    };

    pyoperon.url = "github:heal-research/pyoperon/cpp20";
  };

  outputs = { self, flake-utils, nixpkgs, mach-nix, pypi-deps-db, pyoperon }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        mach = mach-nix.lib.${system};

        pyoperon-pkg = pyoperon.defaultPackage.${system};

        eigency = mach.buildPythonPackage {
          pname = "eigency";
          version = "1.8.0";

          src = builtins.fetchGit {
            url = "https://github.com/wouterboomsma/eigency";
            rev = "205efc54b747f660bfd211e0040d793eed70b0d7";
            submodules = true;
          };

          requirements = ''
            cmake
            cython
            numpy
            setuptools
            '';
        };

        feat = mach.buildPythonPackage {
          pname = "feat_ml";
          version = "0.5.2";
          src = builtins.fetchGit {
            url = "https://github.com/cavalab/feat";
            rev = "0095fc8f9a41db1440f65a0641649fe80621a6ab";
          };

          requirements = ''
            cmake
            cython
            pandas
            pmlb
            setuptools
            scikit-learn
            '';

          nativeBuildInputs = with pkgs; [ ninja eigen eigency shogun ];

          # environment variables
          SHOGUN_LIB="${pkgs.shogun}/lib";
          SHOGUN_DIR="${pkgs.shogun.dev}/include";
          EIGEN3_INCLUDE_DIR="${pkgs.eigen}/include/eigen3";
          LD_LIBRARY_PATH="${placeholder "out"}";
        };

        ellyn = mach.buildPythonPackage {
          pname = "ellyn";
          version = "1.2.0";
          src = builtins.fetchGit {
            url = "https://github.com/cavalab/ellyn";
            rev = "cdff25b2851d942db1cdb2a6796ea61c41396c7c";
          };

          requirements = ''
            numpy
            scikit-learn
            pandas
            DistanceClassifier
            '';

          nativeBuildInputs = with pkgs; [ boost eigen ];
        };

        python = pkgs.python39.override { stdenv = pkgs.gcc11Stdenv; };

      in rec {
        devShell = pkgs.mkShellNoCC {
          name = "SRBench dev";

          buildInputs = with pkgs; [
            python
            (python.withPackages(ps: with ps; [
              pandas
              sympy
              joblib
              pyyaml
              scikit-learn
            ]))
            pyoperon-pkg
            feat
            git-lfs
          ];

          shellHook = ''
            PYTHONPATH=$PYTHONPATH:${pyoperon-pkg}
            LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.gcc11Stdenv.cc.cc.lib python ]}"
            '';
        };
      });
}
