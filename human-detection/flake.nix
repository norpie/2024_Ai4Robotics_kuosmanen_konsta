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

      packages = with pkgs; [
        (python311.withPackages (ps:
          with ps; [
            (opencv4.override {
                enableGtk3 = true;
            })
            matplotlib
            numpy
            moviepy
            ffmpeg
          ]))
      ];
    in {
      devShell = pkgs.mkShell {
        buildInputs = packages;

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath packages}:$LD_LIBRARY_PATH
        '';
      };
    });
}
