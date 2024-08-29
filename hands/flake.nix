{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};

      libraries = with pkgs; [
        stdenv.cc.cc
        stdenv.cc.cc.lib
        libGL
        glib
        libz
        xorg.libSM
      ];

      packages = with pkgs; [
        (python311.withPackages (ps:
          with ps; [
            pip
            (opencv4.override {
                enableGtk3 = true;
            })
          ]))
      ];
    in {
      devShell = pkgs.mkShell {
        buildInputs = packages;

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libraries}:$LD_LIBRARY_PATH
          export LD_PRELOAD=\
          ${pkgs.libGL}/lib/libGL.so:\
          ${pkgs.glib.out}/lib/libgthread-2.0.so.0:\
          ${pkgs.libz.out}/lib/libz.so.1:\
          ${pkgs.xorg.libSM.out}/lib/libSM.so.6:\
          $LD_PRELOAD
          [[ -d ".venv" ]] || python3 -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
        '';
      };
    });
}
