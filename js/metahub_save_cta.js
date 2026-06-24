import { app } from "../../scripts/app.js";

const METAHUB_SAVE_NODE_CLASSES = new Set([
    "MetaHubSaveImage",
    "MetaHubSaveNode",
    "MetaHubSaveVideoNode",
]);

const IMAGE_METAHUB_URL = "https://www.imagemetahub.com/";
const IMAGE_METAHUB_DEEPLINK = "imagemetahub://open";
const HIDE_CTA_SETTING_ID = "ImageMetaHub.SaveNode.hideCTAs";
const CTA_WIDGET_NAMES = new Set([
    "Open in Image MetaHub",
    "Get Image MetaHub",
]);

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

function shouldHideCTA() {
    if (app.extensionManager?.setting?.get) {
        return app.extensionManager.setting.get(HIDE_CTA_SETTING_ID) === true;
    }

    if (app.ui?.settings?.getSettingValue) {
        return app.ui.settings.getSettingValue(HIDE_CTA_SETTING_ID) === true;
    }

    return false;
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

    if (shouldHideCTA()) {
        removeMetaHubCTA(node);
        return;
    }

    if (hasMetaHubCTA(node)) {
        return;
    }

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

function hasMetaHubCTA(node) {
    return node.widgets?.some((widget) => CTA_WIDGET_NAMES.has(widget.name)) === true;
}

function removeMetaHubCTA(node) {
    if (!Array.isArray(node.widgets)) {
        return;
    }

    const nextWidgets = node.widgets.filter(
        (widget) => !CTA_WIDGET_NAMES.has(widget.name)
    );

    if (nextWidgets.length !== node.widgets.length) {
        node.widgets = nextWidgets;
        node.setSize?.(node.computeSize?.() ?? node.size);
    }
}

function syncMetaHubCTA(node) {
    if (!METAHUB_SAVE_NODE_CLASSES.has(node.comfyClass)) {
        return;
    }

    if (shouldHideCTA()) {
        removeMetaHubCTA(node);
    } else {
        addMetaHubCTA(node);
    }
}

function syncAllMetaHubCTAs() {
    for (const node of app.graph?._nodes ?? []) {
        syncMetaHubCTA(node);
    }

    app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "ImageMetaHub.SaveNodeCTA",
    settings: [
        {
            id: HIDE_CTA_SETTING_ID,
            name: "Hide Open/Get Image MetaHub buttons",
            type: "boolean",
            defaultValue: false,
            category: ["Image MetaHub", "Save Node", "Hide Open/Get Image MetaHub buttons"],
            tooltip:
                'Hide the "Open in Image MetaHub" and "Get Image MetaHub" buttons on MetaHub Save nodes.',
            onChange: syncAllMetaHubCTAs,
        },
    ],

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
        syncMetaHubCTA(node);
    },
});
