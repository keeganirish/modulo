# Modulo M icon pack

## Included
- `modulo-m-icon.svg`: scalable app icon
- `modulo-wordmark.svg`: icon + Modulo wordmark for application header
- `modulo.ico` and `favicon.ico`: multi-size Windows icon and browser favicon
- PNG assets at 16, 24, 32, 48, 64, 96, 128, 150, 180, 192, 256, and 512 px
- `apple-touch-icon.png`, `icon-192.png`, `icon-512.png`, `site.webmanifest`

## Palette
- Deep navy / blue background: `#07162F`, `#123D78`
- White-to-blue-to-teal M: `#FFFFFF`, `#83BDFF`, `#20E3C2`

## Browser integration
Put this folder where Flask serves `/icons/`, then add inside `frontend/index.html`'s `<head>`:
```html
<link rel="icon" href="/icons/favicon.ico" sizes="any">
<link rel="icon" type="image/svg+xml" href="/icons/modulo-m-icon.svg">
<link rel="apple-touch-icon" href="/icons/apple-touch-icon.png">
<link rel="manifest" href="/icons/site.webmanifest">
<meta name="theme-color" content="#07162F">
```

## Windows
Use `modulo.ico` for the Windows shortcut or packaged executable. For PyInstaller, configure the build/spec to use this file as the application icon. For a browser-only local app, set the shortcut icon to this ICO; browser window chrome itself is controlled by the browser.

## Header
Use `modulo-wordmark.svg` for a light header. Use `modulo-m-icon.svg` alone for a compact app mark.

All assets are standalone and ready to copy into your app. Adjust `/icons/` URLs if serving from a different path.
