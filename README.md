# langchain-poc

実行例

```console
> Entering new AgentExecutor chain...
Thought: Cloud Run JobsをTerraformで作成するコードを探す
Action: google_serper
Action Input: "Cloud Run Jobs Terraform"
Observation: [{'title': 'DevOps with GCP (Remote)', 'link': 'https://www.google.com/search?gl=jp&oq=Cloud+Run+Jobs+Terraform&hl=ja&q=Cloud+Run+Jobs+Terraform&ibp=htl;jobs&sa=X&ved=2ahUKEwiv4Prx9s2BAxVZKlkFHe5MAXgQkd0GegQIFBAB#fpstate=tldetail&htivrt=jobs&htiq=Cloud+Run+Jobs+Terraform&htidocid=J1C4V8yugN5AnigCAAAAAA%3D%3D', 'company_name': 'RedSpix', 'location': '  アメリカ合衆国 イリノイ州 シカゴ   ', 'via': '提供元: Jooble', 'extensions': ['28 日前', '契約社員'], 'detected_extensions': {'posted_at': '28 日前', 'schedule_type': '契約社員'}}]
Thought: registry.terraform.ioからCloud Run JobsをTerraformで作成するコードを探す
Action: lookup
Action Input: https://registry.terraform.io/
Observation:  The code to create a Cloud Run Job with Terraform would look like this:

resource "google_cloud_run_v2_job" "default" {
  name     = "cloudrun-job"
  location = "us-central1"

  template {
    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello"
    }
  }

  vpc_access {
    connector = google_vpc_access_connector.connector.id
    egress    = "ALL_TRAFFIC"
  }

  lifecycle {
    ignore_changes = [
      launch_stage,
    ]
  }
}

resource "google_vpc_access_connector" "connector" {
  name = "resource_name"
}
Thought: 以上のコードを使用してCloud Run JobsをTerraformで作成する
Final Answer: 以上のコードを使用することで、Cloud Run JobsをTerraformで作成することができます。

> Finished chain.
```
