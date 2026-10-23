import os
import re
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import Mock
from image_trust import images, verify


class ImageTrustTest(unittest.TestCase):
    def test_git_comparison_finds_nested_images_and_ignores_tag_only_changes(self):
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                def git(*args):
                    return subprocess.check_output(["git", "-c", "user.name=Test", "-c",
                        "user.email=test@example.invalid", *args], text=True, stderr=subprocess.DEVNULL)
                git("init")
                path = Path("registry/engines/test.json")
                path.parent.mkdir(parents=True)
                digest = "a" * 64
                path.write_text('{"image":"ghcr.io/0xsero/gateway:old@sha256:' + digest + '"}', encoding="utf-8")
                git("add", ".")
                git("commit", "-m", "base")
                base = git("rev-parse", "HEAD").strip()
                path.write_text('{"image":"ghcr.io/0xsero/gateway:new@sha256:' + digest +
                    '","nested":{"image":"ghcr.io/0xsero/exl3xpu@sha256:' + "b" * 64 + '"}}', encoding="utf-8")
                git("add", ".")
                git("commit", "-m", "new image")
                self.assertEqual(images("HEAD") - images(base),
                    {"ghcr.io/0xsero/exl3xpu@sha256:" + "b" * 64})
                with self.assertRaises(subprocess.CalledProcessError):
                    images("missing-base")
            finally:
                os.chdir(old)

    def identity(self, image):
        run = Mock()
        verify(image, run)
        args, kwargs = run.call_args
        self.assertEqual(args[0][:5], ["gh", "attestation", "verify", "oci://" + image, "--repo"])
        self.assertIn("--deny-self-hosted-runners", args[0])
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["timeout"], 120)
        return args[0][5], re.compile(args[0][args[0].index("--cert-identity-regex") + 1])

    def test_verification_binds_publisher_workflow_and_branch(self):
        for name in ("gateway", "deepseek-v4.1-flash-spark"):
            repo, identity = self.identity(f"ghcr.io/0xsero/{name}@sha256:" + "a" * 64)
            self.assertEqual(repo, "0xSero/local-ai-images")
            base = "https://github.com/0xSero/local-ai-images/.github/workflows/release-image.yml@"
            self.assertTrue(identity.search(base + "refs/heads/main"))
            for ref in ["refs/heads/image/x", "refs/tags/b1", "refs/heads/main2"]:
                self.assertFalse(identity.search(base + ref), ref)
            self.assertFalse(identity.search("https://github.com/0xSero/local-ai-images/.github/workflows/other.yml@refs/heads/main"))

    def test_org_successors_use_only_the_canonical_main_publisher(self):
        for name in ("gateway", "glm53-flash-offload", "dsv41-flash-offload", "exl3xpu"):
            repo, identity = self.identity(f"ghcr.io/sybil-solutions/{name}@sha256:" + "a" * 64)
            self.assertEqual(repo, "sybil-solutions/local-ai-images")
            base = "https://github.com/sybil-solutions/local-ai-images/.github/workflows/"
            self.assertTrue(identity.search(base + "release-image.yml@refs/heads/main"))
            self.assertFalse(identity.search(base + "release-image.yml@refs/heads/feature"))
            self.assertFalse(identity.search(base + "other.yml@refs/heads/main"))
            self.assertFalse(identity.search("https://github.com/sybil-solutions/exl3xpu/.github/workflows/release-image.yml@refs/heads/main"))

    def test_org_gateway_also_accepts_the_multiarch_index_workflow(self):
        _, identity = self.identity("ghcr.io/sybil-solutions/gateway@sha256:" + "a" * 64)
        base = "https://github.com/sybil-solutions/local-ai-images/.github/workflows/"
        for workflow in ("release-image.yml", "multiarch-index.yml"):
            self.assertTrue(identity.search(base + workflow + "@refs/heads/main"), workflow)
            self.assertFalse(identity.search(base + workflow + "@refs/heads/feature"), workflow)
        self.assertFalse(identity.search(base + "multiarch-index.yml.evil@refs/heads/main"))
        for name in ("glm53-flash-offload", "sglang-exl3"):
            _, other = self.identity(f"ghcr.io/sybil-solutions/{name}@sha256:" + "a" * 64)
            self.assertFalse(other.search(base + "multiarch-index.yml@refs/heads/main"), name)

    def test_legacy_proof_comes_from_oci_after_repository_transfer(self):
        for namespace, expected in (("0xsero", True), ("sybil-solutions", False)):
            run = Mock()
            verify(f"ghcr.io/{namespace}/gateway@sha256:" + "a" * 64, run)
            self.assertEqual("--bundle-from-oci" in run.call_args.args[0], expected)

    def test_llamacpp_release_tags_and_master_pass_but_other_refs_do_not(self):
        _, identity = self.identity("ghcr.io/ggml-org/llama.cpp@sha256:" + "a" * 64)
        base = "https://github.com/ggml-org/llama.cpp/.github/workflows/docker.yml@"
        for ref in ["refs/heads/master", "refs/tags/b11146"]:
            self.assertTrue(identity.search(base + ref), ref)
        for ref in ["refs/heads/main", "refs/heads/feature", "refs/tags/b11146-evil", "refs/tags/v1", "refs/heads/master/x"]:
            self.assertFalse(identity.search(base + ref), ref)

    def test_official_vllm_rocm_is_approved_by_image_name(self):
        # images() strips the tag before verify(). A tag left on is not its own approval.
        run = Mock(side_effect=AssertionError("network accessed"))
        verify("vllm/vllm-openai-rocm@sha256:" + "a" * 64, run)
        run.assert_not_called()
        for image in (
            "vllm/vllm-openai-rocm:v0.31.0@sha256:" + "a" * 64,
            "vllm/vllm-openai-rocm:v0.31.0-base@sha256:" + "a" * 64,
            "vllm/vllm-openai-rocm:latest",
            "vllm/vllm-openai-rocm:latest-base",
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                verify(image, run)

    def test_unapproved_or_mutable_images_fail_before_network(self):
        for image in ["ghcr.io/0xsero/gateway:latest", "evil/gateway@sha256:" + "a" * 64]:
            with self.subTest(image=image), self.assertRaises(ValueError):
                verify(image, Mock(side_effect=AssertionError("network accessed")))

    def test_verification_failure_is_fatal(self):
        for error in [subprocess.CalledProcessError(1, "gh"), subprocess.TimeoutExpired("gh", 120)]:
            with self.subTest(error=error), self.assertRaises(type(error)):
                verify("ghcr.io/0xsero/gateway@sha256:" + "a" * 64, Mock(side_effect=error))


if __name__ == "__main__":
    unittest.main()
