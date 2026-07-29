{
  description = "MAI dataset generation and research workbench";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
        python = pkgs.python312;
        runtimeLibraries =
          (with pkgs; [
            libjpeg
            libpng
            openssl
            zlib
          ])
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux (with pkgs; [
            glib
            libGL
            stdenv.cc.cc.lib
          ]);
      in {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cacert
            cmake
            git
            git-lfs
            imagemagick
            ninja
            pkg-config
            python
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibraries;

          shellHook = ''
            export PIP_DISABLE_PIP_VERSION_CHECK=1
            export PYTHONNOUSERSITE=1

            if [[ -d /run/opengl-driver/lib ]]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            fi

            if [[ ! -x .venv/bin/python ]]; then
              echo "Creating Python 3.12 virtual environment in .venv"
              python -m venv .venv
            fi

            source .venv/bin/activate

            echo "MAI development shell"
            echo "  Tests:        python -m pip install -e '.[hub,parquet]'"
            echo "  Full install: ./install.sh"
          '';
        };
      }
    );

    formatter = forAllSystems (
      system: (import nixpkgs {inherit system;}).alejandra
    );

    checks = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in {
        syntax =
          pkgs.runCommand "mai-python-syntax" {
            nativeBuildInputs = [pkgs.python312];
          } ''
            cp -R ${self} source
            chmod -R u+w source
            python -m compileall -q source/mai source/tests
            python -c 'import tomllib; tomllib.load(open("source/pyproject.toml", "rb"))'
            touch "$out"
          '';
      }
    );
  };
}
