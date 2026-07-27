import {
  createRender,
  useModelState,
  useModel,
  useExperimental,
} from "@anywidget/react";
import { Viewer } from "./widget.jsx";

function AnyWidgetApp() {
  const [[data, buffers], _setData] = useModelState("data");
  const experimental = useExperimental();
  const model = useModel();

  return (
    <Viewer
      {...data}
      buffers={buffers}
      experimental={experimental}
      model={model}
    />
  );
}

export default {
  render: createRender(AnyWidgetApp),
};
