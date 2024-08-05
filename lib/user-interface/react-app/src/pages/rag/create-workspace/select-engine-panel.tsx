import {
  Container,
  Header,
  FormField,
  Tiles,
} from "@cloudscape-design/components";

export default function SelectEnginePanel(props: {
  engine: string;
  engines: Map<string, boolean>;
  setEngine: (engine: string) => void;
}) {
  return (
    <Container header={<Header variant="h2">Workspace Engine</Header>}>
      <FormField label="Vector Engine" stretch={true}>
        <Tiles
          items={[
            {
              value: "opensearch",
              label: "Amazon OpenSearch Serverless",
              description:
                "The vector engine for Amazon OpenSearch Serverless introduces a simple, scalable, and high-performing vector storage and search capability.",
              disabled: props.engines.get("opensearch") !== true,
            },
          ]}
          value={props.engine}
          onChange={(e) => props.setEngine(e.detail.value)}
        />
      </FormField>
    </Container>
  );
}
