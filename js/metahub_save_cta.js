import { app } from "../../scripts/app.js";

const METAHUB_SAVE_NODE_CLASSES = new Set([
    "MetaHubSaveImage",
    "MetaHubSaveNode",
    "MetaHubSaveVideoNode",
]);

const IMAGE_METAHUB_URL = "https://www.imagemetahub.com/";
const IMAGE_METAHUB_DEEPLINK = "imagemetahub://open";

function buildDeepLink(node) {
    const savedPath = node.__imageMetaHubLastSavedPath;
    const params = new URLSearchParams({ source: "comfyui-save-node" });

    if (savedPath) {
        params.set("file", savedPath);
    }

    return `${IMAGE_METAHUB_DEEPLINK}?${params.toString()}`;
}

function openUrl(url) {
    window.open(url, "_blank", "noopener,noreferrer");
}

function getSavedPaths(message) {
    const files = message?.imagemetahub?.files;
    if (!Array.isArray(files)) {
        return [];
    }
    return files.filter((filePath) => typeof filePath === "string" && filePath.trim());
}

function addMetaHubCTA(node) {
    if (!METAHUB_SAVE_NODE_CLASSES.has(node.comfyClass)) {
        return;
    }

    if (node.__imageMetaHubCTAAdded) {
        return;
    }

    node.__imageMetaHubCTAAdded = true;
    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        const savedPaths = getSavedPaths(message);
        if (savedPaths.length > 0) {
            this.__imageMetaHubLastSavedPath = savedPaths[savedPaths.length - 1];
        }
    };

    node.addWidget(
        "button",
        "Open in Image MetaHub",
        null,
        () => {
            if (!node.__imageMetaHubLastSavedPath) {
                window.alert("Generate an image with this node first.");
                return;
            }
            openUrl(buildDeepLink(node));
        }
    );

    node.addWidget(
        "button",
        "Get Image MetaHub",
        null,
        () => openUrl(IMAGE_METAHUB_URL)
    );
}

app.registerExtension({
    name: "ImageMetaHub.SaveNodeCTA",

    async nodeCreated(node) {
        addMetaHubCTA(node);
    },
});
