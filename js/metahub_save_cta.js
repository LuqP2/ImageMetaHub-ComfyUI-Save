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
    const candidates = [
        message?.imagemetahub_files,
        message?.ui?.imagemetahub_files,
        message?.imagemetahub?.files,
        message?.ui?.imagemetahub?.files,
    ];

    for (const files of candidates) {
        if (Array.isArray(files)) {
            return files.filter(
                (filePath) => typeof filePath === "string" && filePath.trim()
            );
        }
    }

    return [];
}

function captureLastSavedPath(node, message) {
    const savedPaths = getSavedPaths(message);
    if (savedPaths.length > 0) {
        node.__imageMetaHubLastSavedPath = savedPaths[savedPaths.length - 1];
    }
}

function addMetaHubCTA(node) {
    if (!METAHUB_SAVE_NODE_CLASSES.has(node.comfyClass)) {
        return;
    }

    if (node.__imageMetaHubCTAAdded) {
        return;
    }

    node.__imageMetaHubCTAAdded = true;

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

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!METAHUB_SAVE_NODE_CLASSES.has(nodeData.name)) {
            return;
        }

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalOnExecuted?.apply(this, arguments);
            captureLastSavedPath(this, message);
        };
    },

    async nodeCreated(node) {
        addMetaHubCTA(node);
    },
});
