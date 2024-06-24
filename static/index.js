import * as THREE from 'https://unpkg.com/three@0.127.0/build/three.module.js';
import {OrbitControls} from 'https://unpkg.com/three@0.127.0/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'https://unpkg.com/three@0.127.0/examples/jsm/loaders/GLTFLoader.js';
import {DragControls} from 'https://unpkg.com/three@0.127.0/examples/jsm/controls/DragControls.js'


let scene, camera, renderer, cube, isDrawing, pointerDown, controls;
let frameIndex = 0;
let pointCloudData;
const now = Date.now();
let pointCloud
let quaternionStart, positionStart;
let isGenerating = false;

const allModelsUrl = ['background.glb', 'basin.glb', 'bed.glb', 'flower.glb', 'kitchen_chair_1.glb', 'kitchen_chair_2.glb', 'office_chair.glb', 'sofa.glb', 'table.glb', 'wc.glb'];
// const allModelsUrl = ['table.glb']
const allModels_dict = {}
const allModels_list = []
let initLoc = {}

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    // camera.position.set(9, 6, -9);
    camera.position.set(3, 13, 0);


    // var qm = new THREE.Quaternion(0.1, 0.2, 0.3, 0.4);
    // camera.quaternion.copy(qm)
    // camera.updateMatrixWorld();

    scene.add(camera)

    renderer = new THREE.WebGLRenderer({alpha: true, antialias: true});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    // document.body.appendChild(renderer.domElement);

    const canvasContainer = document.getElementById('canvas-container');
    canvasContainer.appendChild(renderer.domElement);

    const planeGeometry = new THREE.PlaneGeometry(1000, 1000);
    const plane = new THREE.Mesh(planeGeometry, new THREE.MeshBasicMaterial({ color: 0x000000, opacity: 0.25, transparent: true }));
    plane.rotation.x = -Math.PI / 2;
    plane.visible = false; // Make the plane invisible
    scene.add(plane);

    let points = [];
    const line_geometry = new THREE.BufferGeometry().setFromPoints(points);
    const line_material = new THREE.LineBasicMaterial({ color: 0x0000ff, linewidth: 3 });
    const line = new THREE.Line(line_geometry, line_material);
    scene.add(line);


    controls = new OrbitControls(camera, renderer.domElement);
    // controls.addEventListener('change', render);

    // const quaternion = new THREE.Quaternion(-0.707, 0, 0, 0.707)
    // camera.quaternion.copy(quaternion)

    // const light = new THREE.HemisphereLight(0xffffbb, 0x080820, 1);
    // scene.add(light);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6); // soft white light
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, 10, 5   );
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const grid = new THREE.GridHelper(30, 30);
    scene.add(grid);

    // Handle dynamic point cloud
    const geometry = new THREE.BufferGeometry();
    const points_frame = new Float32Array(1048 * 3);
    geometry.setAttribute('position', new THREE.BufferAttribute(points_frame, 3));
    const material = new THREE.PointsMaterial({ size: 0.04, color: 0x2020e0 });
    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);


    for (let i = 0; i < allModelsUrl.length; i++) {
        const assetLoader = new GLTFLoader();
        assetLoader.load('/static/'.concat(allModelsUrl[i]), function (gltf) {
            const model = gltf.scene;
            if (i !== 0) {allModels_dict[allModelsUrl[i]] = model;
                allModels_list.push(model);
                initLoc[allModelsUrl[i]] = model.children[0].position.clone()
            }

            scene.add(model);
        }, undefined, function (error) {
            console.error(error);
        });
    }

    // for (let [key, value] of Object.entries(allModels_dict)) {
    //     initLoc[key] = value.children[0].position
    // }


    const dragControls = new DragControls(allModels_list, camera, renderer.domElement);

    dragControls.addEventListener('dragstart', function (event) {
        if (isDrawing) {return}
        controls.enabled = false;
        raycaster.setFromCamera(mouse, camera);

        const intersects = raycaster.intersectObject(plane);
        if (intersects.length > 0) {
            const point = intersects[0].point;
            event.object.position.set(point.x, 0, point.z);
        }
    });

    dragControls.addEventListener('dragend', function (event) {
        if (isDrawing) {return}
        controls.enabled = true;
    });

    dragControls.addEventListener('drag', function (event) {
        if (isDrawing) {return}
        raycaster.setFromCamera(mouse, camera);

        const intersects = raycaster.intersectObject(plane);
        if (intersects.length > 0) {
            const point = intersects[0].point;
            event.object.position.set(point.x, 0, point.z);
        }
    });

    // Raycaster for constraining movement to the plane
    const raycaster = new THREE.Raycaster();

    // Mouse vector for the raycaster
    const mouse = new THREE.Vector2();
    // Update the mouse vector with the current mouse position
    document.addEventListener('pointermove', function (event) {

        const rect = renderer.domElement.getBoundingClientRect();

        mouse.x = ((event.x - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.y - rect.top) / rect.height) * 2 + 1;

        if (isDrawing && pointerDown && points.length < 500) {
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(plane);
            const point = intersects[0].point;
            point.y = 0.1
            points.push(point);
            line.geometry.setFromPoints(points);
        }}, false);

    // pointerDown = false;
    document.addEventListener('pointerdown', function (event) {
        pointerDown = true;
    }, false);
    document.addEventListener('pointerup', function (event) {
        pointerDown = false;
        isDrawing = false;
        controls.enabled = true
        dragControls.enabled = true
        document.getElementById('toggleDraw').innerHTML = "Draw Trajectory";

    }, false);

    document.getElementById('toggleDraw').addEventListener('click', toggleDrawing);
    isDrawing = false;


    function toggleDrawing() {
        controls.enabled = false
        quaternionStart = camera.quaternion.clone();
        positionStart = camera.position.clone();
        points = []
        isDrawing = true;

        dragControls.enabled = false

        document.getElementById('toggleDraw').innerHTML = "Drawing";
    }

    document.getElementById('reset').addEventListener('click', reset);

    function reset() {
        points = [];
        line_geometry.setFromPoints(points);

        for (let [key, value] of Object.entries(allModels_dict)) {
            value.children[0].position.set(initLoc[key].x, initLoc[key].y, initLoc[key].z);
        }
        document.getElementById('toggleDraw').innerHTML = "Draw Trajectory";
        pointCloudData = null;
        pointCloud.geometry.attributes.position.array = new Float32Array(1048 * 3);
        pointCloud.geometry.attributes.position.needsUpdate = true;
        render()
    }

    document.getElementById('generateMotion').addEventListener('click', runGeneration);

    function runGeneration() {
        if (!points.length || isGenerating) {return}
        isGenerating = true;

        const userData = {}
        for (let [key, value] of Object.entries(allModels_dict)) {
            userData[key] = value.children[0].position
        }
        userData['trajectory'] = points

        fetch('/move_cube', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData),
        })
        .then(response => response.json())
        .then(data => {
            pointCloudData = data
            document.getElementById("generateMotion").innerHTML = "Generate Motion";
            isGenerating = false;
        })
        .catch((error) => {
            console.error('Error:', error);
        });

    }



    document.addEventListener('keypress', function(event) {
        const key = event.key; // "a", "1", "Shift", etc.
        if (key === 'q') {if (isDrawing) {isDrawing = false;}
            else {isDrawing = true;}}
    });

    // Resize Listener
    window.addEventListener('resize', onWindowResize, false);

    animate();
}


function updatePointCloud() {
    if (!pointCloudData) {

        return;
    }
    const positions = pointCloud.geometry.attributes.position.array;
    const frameData = pointCloudData[frameIndex];

    for (let i = 0, j = 0; i < frameData.length; i++, j += 3) {
        positions[j] = frameData[i][0];
        positions[j + 1] = frameData[i][1];
        positions[j + 2] = frameData[i][2];
    }
    pointCloud.geometry.attributes.position.needsUpdate = true;

    frameIndex += 1;
    if (frameIndex >= pointCloudData.length) frameIndex = 0; // Loop back to the start
}

function animate() {
    // requestAnimationFrame(animate);
    setTimeout(() => {
    requestAnimationFrame(animate);
  }, 1000 / 25);
    updatePointCloud()

    if (isGenerating) {
        let dotNum = parseInt((Date.now() - now) / 500) % 8
        const dot = Array(dotNum * 2).join('.')
        document.getElementById("generateMotion").innerHTML = "Loading" + dot;
    }

    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);

    render();
}


// function onDocumentKeyDown(event) {
//     var keyCode = event.which;
//     // Move the cube with arrow keys
//     if (keyCode == 87) { cube.position.y += 0.1; } // W key
//     if (keyCode == 83) { cube.position.y -= 0.1; } // S key
//     if (keyCode == 65) { cube.position.x -= 0.1; } // A key
//     if (keyCode == 68) { cube.position.x += 0.1; } // D key
//     sendCubePosition();
// }

// function runGeneration() {
//     const position = {
//         x: cube.position.x,
//         y: cube.position.y,
//         z: cube.position.z
//     };
//
//     fetch('/move_cube', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify(position),
//     })
//     .then(response => response.json())
//     .then(data => {
//         // Use the data to create a new cube
//         const newGeometry = new THREE.BoxGeometry();
//         const newMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
//         const newCube = new THREE.Mesh(newGeometry, newMaterial);
//         newCube.position.set(data.x, data.y, data.z);
//         scene.add(newCube);
//     })
//     .catch((error) => {
//         console.error('Error:', error);
//     });
// }

function render() {
    renderer.render(scene, camera);
}

// document.addEventListener("keydown", onDocumentKeyDown, false);
init();