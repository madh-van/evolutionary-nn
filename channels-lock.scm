(list (channel
        (name 'non-guix)
        (url "https://gitlab.com/nonguix/nonguix")
        (commit
          "5b1184a103640186fd22e1c53ba64a83f7e6fc3d"))
      (channel
        (name 'guix)
        (url "https://git.savannah.gnu.org/git/guix.git")
        (commit
          "37186ec462dc574477af195aa07fea3fe1ed07a2")
        (introduction
          (make-channel-introduction
            "9edb3f66fd807b096b48283debdcddccfea34bad"
            (openpgp-fingerprint
              "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA")))))
