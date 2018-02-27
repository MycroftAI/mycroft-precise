# -*- mode: python -*-
block_cipher = None

a = Analysis(['precise/scripts/engine.py'],
             pathex=['.'],
             binaries=[],
             datas=[],
             hiddenimports=['prettyparse', 'speechpy'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PySide', 'PyQt4', 'PyQt5', 'matplotlib'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='precise-engine',
          debug=False,
          strip=True,
          upx=True,
          console=True,)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=True,
               upx=True,
               name='precise-engine')
