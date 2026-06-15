import { app } from "../../scripts/app.js";

const METAHUB_SAVE_NODE_CLASSES = new Set([
    "MetaHubSaveImage",
    "MetaHubSaveNode",
    "MetaHubSaveVideoNode",
]);

const IMAGE_METAHUB_URL = "https://www.imagemetahub.com/";
const IMAGE_METAHUB_DEEPLINK = "imagemetahub://open";

function getWidgetValue(node, widgetName) {
    const widget = node.widgets?.find((candidate) => candidate.name === widgetName);
    const value = widget?.value;
    return typeof value === "string" ? value.trim() : "";
}

function buildDeepLink(node) {
    const outputPath = getWidgetValue(node, "output_path");
    const params = new URLSearchParams({ source: "comfyui-save-node" });

    if (outputPath) {
        params.set("path", outputPath);
    }

    return `${IMAGE_METAHUB_DEEPLINK}?${params.toString()}`;
}

function openUrl(url) {
    window.open(url, "_blank", "noopener,noreferrer");
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
        () => openUrl(buildDeepLink(node))
    );

    node.addWidget(
        "button",
        "Download Image MetaHub",
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
