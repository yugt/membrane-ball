# Google Play Store Mobile Packaging Guide

This guide outlines how to package the high-performance **Three.js WebGL game (`game_web/`)** into a native Android Application Package (`.apk` or `.aab`) using **Ionic Capacitor** to publish it on the Google Play Store.

---

## 1. Prerequisites
Before packaging, make sure you have the following installed on your developer machine:
1. **Node.js** (v18 or higher) — [Download Node.js](https://nodejs.org/)
2. **Android Studio** (with Android SDK and Gradle) — [Download Android Studio](https://developer.android.com/studio)

---

## 2. Step-by-Step Capacitor Packaging

Capacitor is a modern, high-performance hybrid wrapper that embeds your web files into a native Android web frame (WebView) with 100% native performance and direct API bridge access.

### Step 2.1: Initialize Node Project
Inside the `game_web/` directory, initialize a standard Node package:
```bash
cd game_web
npm init -y
```

### Step 2.2: Install Capacitor Core and CLI
Install Capacitor as developer dependencies:
```bash
npm install @capacitor/core @capacitor/cli
```

### Step 2.3: Initialize Capacitor Configuration
Initialize Capacitor with your app name and a unique package ID (reverse-domain notation, which is required by the Play Store):
```bash
npx cap init "Membrane Breakout 3D" "com.yourdomain.membranebreakout" --web-dir=.
```
*Note: `--web-dir=.` tells Capacitor that your index.html and assets are located directly in the root of the `game_web/` folder.*

### Step 2.4: Add Android Platform Support
Install the native Android package and add it to your project:
```bash
npm install @capacitor/android
npx cap add android
```
This command creates a fully native Android Gradle project in a new `/android` subfolder inside `game_web/`.

### Step 2.5: Sync and Copy Assets
Whenever you edit your HTML, CSS, or JS files, sync those changes into the Android project shell:
```bash
npx cap copy
```

---

## 3. Building the Native APK / App Bundle

### Step 3.1: Open Android Studio
Open the generated Android project directly in Android Studio:
```bash
npx cap open android
```
Android Studio will open the project, download the required Gradle build tools, and sync the project dependencies.

### Step 3.2: Configure for Production
Inside Android Studio:
1. Open `app/src/main/res/values/styles.xml` and ensure the splash theme and colors match your brand.
2. Replace the default launcher icons in `app/src/main/res/mipmap-*` folders with your own high-resolution game icons (512x512 PNG).
3. Open `/android/app/build.gradle` and verify your `versionCode` (e.g. `1`) and `versionName` (e.g. `"1.0.0"`). Increment these numbers for future store updates!

### Step 3.3: Compile the Release `.aab` / `.apk`
1. Go to the top menu bar and select **Build > Generate Signed Bundle / APK...**
2. Select **Android App Bundle (`.aab`)** (this is the modern format required for all new Google Play uploads) and click Next.
3. Create a new Keystore file (which will act as your digital signature to secure your app). Save this file in a secure location!
4. Choose **release** build variant and signature schemes.
5. Click **Finish**. Android Studio will compile and output your production-ready `.aab` file!

---

## 4. WebView Performance Tuning
We have already optimized `game.js` to ensure extremely high performance on mobile devices.
* **Device Pixel Ratio (DPR) Clamping:** High-resolution mobile screens have a DPR of 3.0+. Rendering WebGL at full resolution on budget mobile devices causes GPU bottlenecks. `game.js` limits this to `Math.min(window.devicePixelRatio, 2)` to ensure a locked **60 FPS** experience.
* **Direct Touch Handling:** The event listener uses native `touchmove` with `{ passive: false }` to block browser screen scrolling, turning touch slide movements into native-feeling game controls.
