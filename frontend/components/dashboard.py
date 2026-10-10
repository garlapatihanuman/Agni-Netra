from __future__ import annotations

import time
from typing import Any, Dict

import streamlit as st


# ---------------------------------------------------------
# Interactive 3D WebGL / Three.js Visualizer Component
# ---------------------------------------------------------
def render_3d_hologram_core(
    hazard_level: str = "STANDBY",
    active_classes: str = "",
):
    """Renders a secondary 3D visualization of the latest detector state."""
    hazard_code = 0
    if hazard_level == "CRITICAL":
        hazard_code = 2
    elif hazard_level in ("WARNING", "ADVISORY"):
        hazard_code = 1
    elif hazard_level == "CLEAR":
        hazard_code = 0
    if "Smoke" in active_classes and "Fire" not in active_classes and "Live Flame" not in active_classes:
        hazard_code = 3

    status_class = {
        0: "hud-clear",
        1: "hud-warn",
        2: "hud-crit",
        3: "hud-warn",
    }[hazard_code]
    class_summary = active_classes or "No classes reported"

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                overflow: hidden;
                background: transparent;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                user-select: none;
            }}
            #canvas-container {{
                width: 100%;
                height: 220px;
                position: relative;
                border-radius: 12px;
                background: radial-gradient(circle at center, rgba(22, 34, 58, 0.45) 0%, rgba(8, 12, 22, 0.96) 100%);
                border: 1px solid rgba(157, 178, 206,  0.2);
                box-shadow: inset 0 0 32px rgba(0, 0, 0, 0.7);
                overflow: hidden;
            }}
            /* Ambient Cyber Grid Floor */
            .cyber-grid {{
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 45%;
                background: linear-gradient(180deg, transparent 0%, rgba(255, 106, 0, 0.05) 100%),
                            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px),
                            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px);
                background-size: 100% 100%, 24px 24px, 24px 24px;
                pointer-events: none;
                opacity: 0.6;
            }}
            /* Tactical Telemetry HUD */
            .hud-overlay-top {{
                position: absolute;
                top: 14px;
                left: 18px;
                color: #00f0ff;
                font-family: 'JetBrains Mono', monospace, Consolas;
                font-size: 11px;
                letter-spacing: 1.5px;
                text-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
                pointer-events: none;
                z-index: 10;
            }}
            .hud-radar-reticle {{
                position: absolute;
                top: 14px;
                right: 18px;
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                letter-spacing: 1px;
                color: #9db2ce;
                pointer-events: none;
                z-index: 10;
            }}
            .radar-pulse-ring {{
                width: 14px;
                height: 14px;
                border-radius: 50%;
                border: 1.5px solid #00f0ff;
                animation: radarRing 2s infinite linear;
                position: relative;
            }}
            @keyframes radarRing {{
                0% {{ transform: scale(0.6); opacity: 1; }}
                100% {{ transform: scale(1.6); opacity: 0; }}
            }}
            .hud-status {{
                position: absolute;
                bottom: 14px;
                right: 18px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                letter-spacing: 1.5px;
                pointer-events: none;
                z-index: 10;
                padding: 6px 14px;
                border-radius: 8px;
                font-weight: 700;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
            }}
            .hud-clear {{ color: #00e676; border: 1px solid rgba(0, 230, 118, 0.5); background: rgba(0, 230, 118, 0.12); box-shadow: 0 0 15px rgba(0,230,118,0.2); }}
            .hud-warn {{ color: #ffaa00; border: 1px solid rgba(255, 170, 0, 0.6); background: rgba(255, 170, 0, 0.15); box-shadow: 0 0 20px rgba(255,170,0,0.3); }}
            .hud-crit {{ color: #ff1744; border: 1px solid rgba(255, 23, 68, 0.7); background: rgba(255, 23, 68, 0.2); box-shadow: 0 0 25px rgba(255,23,68,0.45); }}
            
            .drag-hint {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: rgba(255, 255, 255, 0.25);
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                letter-spacing: 2px;
                pointer-events: none;
                transition: opacity 0.5s ease;
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div class="cyber-grid"></div>

            <div class="hud-overlay-top">
                <div style="font-weight: 700;">SYSTEM VISUALIZATION · 3D PARTICLE FIELD</div>
                <div style="color: #9db2ce; margin-top: 3px;">LATEST DETECTOR STATE: {hazard_level} | CLASSES: {class_summary}</div>
            </div>

            <div class="hud-radar-reticle">
                <div class="radar-pulse-ring"></div>
                <span>ANIMATED SYSTEM VIEW</span>
            </div>
            
            <div id="drag-hint-el" class="drag-hint">DRAG TO ORBIT 360°</div>

            <div id="status-tag" class="hud-status {status_class}">
                DETECTOR STATE: {hazard_level}
            </div>
        </div>

        <script>
            let scene, camera, renderer, flameParticles, smokeParticles, rings = [], coreLight;
            let currentMode = {hazard_code};
            const container = document.getElementById('canvas-container');
            const dragHint = document.getElementById('drag-hint-el');

            function init() {{
                scene = new THREE.Scene();
                camera = new THREE.PerspectiveCamera(48, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.z = 82;

                renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                container.appendChild(renderer.domElement);

                // Central Dynamic Light
                coreLight = new THREE.PointLight(0xff6a00, 2.5, 120);
                coreLight.position.set(0, 0, 0);
                scene.add(coreLight);

                // Build 3D Holographic Orbit Rings with Dashed Styling
                const ringRadii = [21, 28, 35];
                for(let r = 0; r < ringRadii.length; r++) {{
                    const ringGeo = new THREE.RingGeometry(ringRadii[r], ringRadii[r] + 0.45, 64);
                    const ringColor = r === 0 ? 0xff5500 : (r === 1 ? 0x00f0ff : 0xffaa00);
                    const ringMat = new THREE.MeshBasicMaterial({{
                        color: ringColor,
                        side: THREE.DoubleSide,
                        transparent: true,
                        opacity: 0.38
                    }});
                    const ring = new THREE.Mesh(ringGeo, ringMat);
                    ring.rotation.x = Math.PI / 2 + (r * 0.35);
                    ring.rotation.y = r * 0.45;
                    rings.push(ring);
                    scene.add(ring);
                }}

                // Build 1,800 Flame & Ember Fluid Particles
                const flameCount = 1800;
                const flameGeo = new THREE.BufferGeometry();
                const flamePos = new Float32Array(flameCount * 3);
                const flameColors = new Float32Array(flameCount * 3);
                const flameSpeeds = new Float32Array(flameCount);

                for(let i = 0; i < flameCount; i++) {{
                    const theta = Math.random() * Math.PI * 2;
                    const radius = Math.random() * 15;
                    const y = (Math.random() - 0.5) * 36;
                    
                    flamePos[i * 3] = Math.cos(theta) * radius;
                    flamePos[i * 3 + 1] = y;
                    flamePos[i * 3 + 2] = Math.sin(theta) * radius;

                    flameSpeeds[i] = 0.25 + Math.random() * 0.65;
                }}

                flameGeo.setAttribute('position', new THREE.BufferAttribute(flamePos, 3));
                flameGeo.setAttribute('color', new THREE.BufferAttribute(flameColors, 3));
                
                const flameMat = new THREE.PointsMaterial({{
                    size: 2.0,
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.9,
                    blending: THREE.AdditiveBlending
                }});

                flameParticles = new THREE.Points(flameGeo, flameMat);
                scene.add(flameParticles);

                // Build 600 Volumetric Smoke Particles
                const smokeCount = 600;
                const smokeGeo = new THREE.BufferGeometry();
                const smokePos = new Float32Array(smokeCount * 3);
                const smokeColors = new Float32Array(smokeCount * 3);

                for(let i = 0; i < smokeCount; i++) {{
                    const theta = Math.random() * Math.PI * 2;
                    const radius = 8 + Math.random() * 26;
                    const y = Math.random() * 42 - 6;
                    
                    smokePos[i * 3] = Math.cos(theta) * radius;
                    smokePos[i * 3 + 1] = y;
                    smokePos[i * 3 + 2] = Math.sin(theta) * radius;

                    smokeColors[i * 3] = 0.45;
                    smokeColors[i * 3 + 1] = 0.52;
                    smokeColors[i * 3 + 2] = 0.62;
                }}

                smokeGeo.setAttribute('position', new THREE.BufferAttribute(smokePos, 3));
                smokeGeo.setAttribute('color', new THREE.BufferAttribute(smokeColors, 3));

                const smokeMat = new THREE.PointsMaterial({{
                    size: 3.6,
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.32,
                    blending: THREE.NormalBlending
                }});

                smokeParticles = new THREE.Points(smokeGeo, smokeMat);
                scene.add(smokeParticles);

                // Interactive Mouse Rotation
                let isDragging = false, prevMouseX = 0, prevMouseY = 0;
                renderer.domElement.addEventListener('mousedown', (e) => {{
                    isDragging = true;
                    prevMouseX = e.clientX;
                    prevMouseY = e.clientY;
                    if(dragHint) dragHint.style.opacity = '0';
                }});
                window.addEventListener('mouseup', () => isDragging = false);
                window.addEventListener('mousemove', (e) => {{
                    if(isDragging) {{
                        const deltaX = e.clientX - prevMouseX;
                        const deltaY = e.clientY - prevMouseY;
                        scene.rotation.y += deltaX * 0.009;
                        scene.rotation.x += deltaY * 0.009;
                        prevMouseX = e.clientX;
                        prevMouseY = e.clientY;
                    }}
                }});

                // Touch Controls for Mobile / Tablets
                renderer.domElement.addEventListener('touchstart', (e) => {{
                    if(e.touches.length === 1) {{
                        isDragging = true;
                        prevMouseX = e.touches[0].clientX;
                        prevMouseY = e.touches[0].clientY;
                        if(dragHint) dragHint.style.opacity = '0';
                    }}
                }});
                window.addEventListener('touchend', () => isDragging = false);
                window.addEventListener('touchmove', (e) => {{
                    if(isDragging && e.touches.length === 1) {{
                        const deltaX = e.touches[0].clientX - prevMouseX;
                        const deltaY = e.touches[0].clientY - prevMouseY;
                        scene.rotation.y += deltaX * 0.01;
                        scene.rotation.x += deltaY * 0.01;
                        prevMouseX = e.touches[0].clientX;
                        prevMouseY = e.touches[0].clientY;
                    }}
                }});

                window.addEventListener('resize', onWindowResize);
                applyModeVisuals();
                animate();
            }}

            function applyModeVisuals() {{
                const colors = flameParticles.geometry.attributes.color.array;
                const count = colors.length / 3;

                for(let i = 0; i < count; i++) {{
                    if(currentMode === 0) {{
                        // Surveillance Grid: Laser Cyan & Emerald Safe
                        colors[i * 3] = 0.0;
                        colors[i * 3 + 1] = 0.7 + Math.random() * 0.3;
                        colors[i * 3 + 2] = 0.95;
                    }} else if(currentMode === 1) {{
                        // Live Flame: Golden Amber & Hot Yellow-White Core
                        const rand = Math.random();
                        colors[i * 3] = 1.0;
                        colors[i * 3 + 1] = 0.55 + rand * 0.45;
                        colors[i * 3 + 2] = rand * 0.15;
                    }} else if(currentMode === 2) {{
                        // Critical Inferno: Fiery Crimson & Molten Blast
                        const rand = Math.random();
                        colors[i * 3] = 1.0;
                        colors[i * 3 + 1] = 0.1 + rand * 0.35;
                        colors[i * 3 + 2] = rand * 0.1;
                    }} else if(currentMode === 3) {{
                        // Smoke Plume
                        const gray = 0.55 + Math.random() * 0.3;
                        colors[i * 3] = gray;
                        colors[i * 3 + 1] = gray + 0.05;
                        colors[i * 3 + 2] = gray + 0.1;
                    }}
                }}
                flameParticles.geometry.attributes.color.needsUpdate = true;

                if(coreLight) {{
                    if(currentMode === 0) coreLight.color.setHex(0x00f0ff);
                    else if(currentMode === 1) coreLight.color.setHex(0xffaa00);
                    else if(currentMode === 2) coreLight.color.setHex(0xff1744);
                    else coreLight.color.setHex(0x8899aa);
                }}
            }}

            function onWindowResize() {{
                if(!container || !renderer || !camera) return;
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }}

            function animate() {{
                requestAnimationFrame(animate);

                // Spin Rings
                const ringSpeed = currentMode === 2 ? 2.5 : (currentMode === 1 ? 1.5 : 1.0);
                rings.forEach((r, idx) => {{
                    r.rotation.z += (0.004 + idx * 0.003) * ringSpeed;
                }});

                // Dynamic Particle Physics
                const positions = flameParticles.geometry.attributes.position.array;
                const count = positions.length / 3;
                const speedMult = currentMode === 2 ? 2.0 : (currentMode === 1 ? 1.4 : 0.85);

                for(let i = 0; i < count; i++) {{
                    positions[i * 3 + 1] += 0.38 * speedMult; // Rise vertically
                    
                    // Swirl vortex mathematics
                    const x = positions[i * 3];
                    const z = positions[i * 3 + 2];
                    const angle = 0.022 * speedMult;
                    positions[i * 3] = x * Math.cos(angle) - z * Math.sin(angle);
                    positions[i * 3 + 2] = x * Math.sin(angle) + z * Math.cos(angle);

                    // Reset when top boundary reached
                    if(positions[i * 3 + 1] > 26) {{
                        positions[i * 3 + 1] = -22;
                        const theta = Math.random() * Math.PI * 2;
                        const rad = Math.random() * (currentMode === 2 ? 18 : 11);
                        positions[i * 3] = Math.cos(theta) * rad;
                        positions[i * 3 + 2] = Math.sin(theta) * rad;
                    }}
                }}
                flameParticles.geometry.attributes.position.needsUpdate = true;

                // Drift Smoke
                const sPos = smokeParticles.geometry.attributes.position.array;
                const sCount = sPos.length / 3;
                for(let i = 0; i < sCount; i++) {{
                    sPos[i * 3 + 1] += 0.16;
                    if(sPos[i * 3 + 1] > 32) {{
                        sPos[i * 3 + 1] = -18;
                    }}
                }}
                smokeParticles.geometry.attributes.position.needsUpdate = true;

                scene.rotation.y += 0.0025;
                renderer.render(scene, camera);
            }}

            init();
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=235, scrolling=False)


# ---------------------------------------------------------
# Web Audio API Siren Alarm
# ---------------------------------------------------------
_last_siren_time = 0.0


def render_audio_siren(active: bool):
    """Triggers an emergency audio alert via browser Web Audio API when hazard is active."""
    global _last_siren_time
    if not active:
        return
    now = time.time()
    if now - _last_siren_time < 3.5:
        return
    _last_siren_time = now

    alarm_script = """
    <script>
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        
        const now = audioCtx.currentTime;
        osc.frequency.setValueAtTime(650, now);
        osc.frequency.exponentialRampToValueAtTime(950, now + 0.18);
        osc.frequency.exponentialRampToValueAtTime(650, now + 0.36);

        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.45);
    } catch(e) {}
    </script>
    """
    st.components.v1.html(alarm_script, height=0)


# ---------------------------------------------------------
# UI Component: Real-Time Threat Level Banner
# ---------------------------------------------------------
def render_hazard_banner(stats: Dict[str, Any], enable_audio: bool = True):
    """Renders a futuristic real-time Threat Level HUD banner with dynamic glow."""
    hazard = stats.get("hazard_level", "CLEAR")
    total = stats.get("total_detections", 0)
    conf = stats.get("highest_confidence", 0.0)
    counts = stats.get("class_counts", {})
    breakdown_str = ", ".join([f"{c}: {n}" for c, n in counts.items()]) if counts else "None"

    if hazard == "CRITICAL":
        banner_class = "hazard-critical"
        title = "CRITICAL HAZARD DETECTED — LIVE FLAME / ACTIVE FIRE CONFIRMED"
        color = "#ff1744"
        if enable_audio:
            render_audio_siren(True)
    elif hazard == "WARNING":
        banner_class = "hazard-warning"
        title = "HAZARD ADVISORY — FLAME OR SMOKE SIGNATURE IDENTIFIED"
        color = "#ffaa00"
    else:
        banner_class = "hazard-clear"
        title = "ALL CLEAR — SURVEILLANCE ARMED & AIRSPACE SECURE"
        color = "#00e676"

    html = f"""
    <div class="hazard-banner {banner_class}">
        <div>
            <div style="font-size: 1.05rem; font-weight: 800; letter-spacing: 1.2px;">{title}</div>
            <div style="font-size: 0.82rem; opacity: 0.9; margin-top: 5px; font-family: 'Space Grotesk', sans-serif;">
                ACTIVE TARGETS: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #fff;">{total}</span> | 
                PEAK CONFIDENCE: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #fff;">{conf:.1%}</span> | 
                TARGET PROFILE: <span style="font-family: 'JetBrains Mono', monospace; color: #fff;">{breakdown_str}</span>
            </div>
        </div>
        <div style="text-align: right;">
            <span style="font-family: 'Orbitron', monospace; font-size: 0.82rem; font-weight: 800; background: rgba(0,0,0,0.55); padding: 6px 14px; border-radius: 8px; border: 1.5px solid {color}; letter-spacing: 1.5px;">
                LEVEL: {hazard}
            </span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------
# UI Component: Tactical Telemetry Metric Cards
# ---------------------------------------------------------
def render_telemetry_metrics(stats: Dict[str, Any]):
    """Renders sleek glassmorphic tactical telemetry metric cards."""
    total = stats.get("total_detections", 0)
    flames = stats.get("flame_count", 0)
    fires = stats.get("fire_count", 0)
    smokes = stats.get("smoke_count", 0)
    latency = stats.get("inference_time_ms", 0.0)
    highest_conf = stats.get("highest_confidence", 0.0)

    html = f"""
    <div class="telemetry-grid">
        <div class="telemetry-card" style="border-left: 3px solid #ff6a00;">
            <div class="telemetry-label">Live Flames</div>
            <div class="telemetry-value" style="color: {'#ff8c00' if flames > 0 else '#ffffff'};">{flames}</div>
            <div class="telemetry-subtext">{f"{flames} active targets" if flames > 0 else "0 detected"}</div>
        </div>
        <div class="telemetry-card" style="border-left: 3px solid #ff1744;">
            <div class="telemetry-label">Fire Outbreaks</div>
            <div class="telemetry-value" style="color: {'#ff1744' if fires > 0 else '#ffffff'};">{fires}</div>
            <div class="telemetry-subtext">{f"{fires} active blazes" if fires > 0 else "0 detected"}</div>
        </div>
        <div class="telemetry-card" style="border-left: 3px solid #00f0ff;">
            <div class="telemetry-label">Smoke Plumes</div>
            <div class="telemetry-value" style="color: {'#00f0ff' if smokes > 0 else '#ffffff'};">{smokes}</div>
            <div class="telemetry-subtext">{f"{smokes} dispersed clouds" if smokes > 0 else "0 detected"}</div>
        </div>
        <div class="telemetry-card" style="border-left: 3px solid #ffb703;">
            <div class="telemetry-label">Peak Confidence</div>
            <div class="telemetry-value" style="color: {'#ffb703' if total > 0 else '#8b9bb4'};">{f"{highest_conf:.1%}" if total > 0 else "0.0%"}</div>
            <div class="telemetry-subtext">Neural Confidence</div>
        </div>
        <div class="telemetry-card" style="border-left: 3px solid #00e676;">
            <div class="telemetry-label">Engine Latency</div>
            <div class="telemetry-value" style="color: #00e676;">{latency:.0f} <span style="font-size: 1rem; font-weight: 500;">ms</span></div>
            <div class="telemetry-subtext">YOLOv8 + Chromatic</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


