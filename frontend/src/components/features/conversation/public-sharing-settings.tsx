import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { Switch } from "#/components/ui/switch";
import { Button } from "#/components/ui/button";
import { Input } from "#/components/ui/input";
import { Label } from "#/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "#/components/ui/card";
import { Alert, AlertDescription } from "#/components/ui/alert";
import { Copy, ExternalLink, Globe, Lock, AlertTriangle } from "lucide-react";
import { usePublicSharing } from "#/hooks/query/use-public-conversation-sharing";
import { useUpdatePublicSharing } from "#/hooks/mutation/use-public-conversation-sharing";

interface PublicSharingSettingsProps {
  conversationId: string;
}

export function PublicSharingSettings({ conversationId }: PublicSharingSettingsProps) {
  const { t } = useTranslation();
  const [showWarning, setShowWarning] = useState(false);

  const { data: sharingStatus, isLoading } = usePublicSharing(conversationId);
  const updateSharingMutation = useUpdatePublicSharing();

  const handleTogglePublic = async (isPublic: boolean) => {
    if (isPublic && !showWarning) {
      setShowWarning(true);
      return;
    }

    try {
      await updateSharingMutation.mutateAsync({
        conversationId,
        data: { is_public: isPublic },
      });

      toast.success(
        isPublic
          ? t("CONVERSATION_SHARING.MADE_PUBLIC_SUCCESS")
          : t("CONVERSATION_SHARING.MADE_PRIVATE_SUCCESS")
      );

      setShowWarning(false);
    } catch (error) {
      toast.error(
        t("CONVERSATION_SHARING.UPDATE_ERROR", {
          error: error instanceof Error ? error.message : "Unknown error"
        })
      );
    }
  };

  const copyShareUrl = () => {
    if (sharingStatus?.share_url) {
      const fullUrl = `${window.location.origin}${sharingStatus.share_url}`;
      navigator.clipboard.writeText(fullUrl);
      toast.success(t("CONVERSATION_SHARING.URL_COPIED"));
    }
  };

  const copyShareToken = () => {
    if (sharingStatus?.share_token) {
      navigator.clipboard.writeText(sharingStatus.share_token);
      toast.success(t("CONVERSATION_SHARING.TOKEN_COPIED"));
    }
  };

  const openPublicView = () => {
    if (sharingStatus?.share_url) {
      const fullUrl = `${window.location.origin}${sharingStatus.share_url}`;
      window.open(fullUrl, "_blank");
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Globe className="h-5 w-5" />
            {t("CONVERSATION_SHARING.TITLE")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {sharingStatus?.is_public ? (
            <Globe className="h-5 w-5 text-green-600" />
          ) : (
            <Lock className="h-5 w-5 text-gray-600" />
          )}
          {t("CONVERSATION_SHARING.TITLE")}
        </CardTitle>
        <CardDescription>
          {t("CONVERSATION_SHARING.DESCRIPTION")}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Privacy Warning */}
        {showWarning && (
          <Alert className="border-orange-200 bg-orange-50">
            <AlertTriangle className="h-4 w-4 text-orange-600" />
            <AlertDescription className="text-orange-800">
              <div className="space-y-2">
                <p className="font-medium">{t("CONVERSATION_SHARING.WARNING_TITLE")}</p>
                <p>{t("CONVERSATION_SHARING.WARNING_MESSAGE")}</p>
                <div className="flex gap-2 mt-3">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setShowWarning(false)}
                  >
                    {t("CONVERSATION_SHARING.CANCEL")}
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => handleTogglePublic(true)}
                    disabled={updateSharingMutation.isPending}
                  >
                    {t("CONVERSATION_SHARING.CONFIRM_PUBLIC")}
                  </Button>
                </div>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* Toggle Switch */}
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <Label htmlFor="public-toggle" className="text-sm font-medium">
              {t("CONVERSATION_SHARING.MAKE_PUBLIC")}
            </Label>
            <p className="text-xs text-gray-600">
              {sharingStatus?.is_public
                ? t("CONVERSATION_SHARING.PUBLIC_STATUS")
                : t("CONVERSATION_SHARING.PRIVATE_STATUS")
              }
            </p>
          </div>
          <Switch
            id="public-toggle"
            checked={sharingStatus?.is_public || false}
            onCheckedChange={handleTogglePublic}
            disabled={updateSharingMutation.isPending || showWarning}
          />
        </div>

        {/* Share URL Section */}
        {sharingStatus?.is_public && sharingStatus.share_url && (
          <div className="space-y-3 pt-4 border-t">
            <Label className="text-sm font-medium">
              {t("CONVERSATION_SHARING.SHARE_URL")}
            </Label>
            <div className="flex gap-2">
              <Input
                value={`${window.location.origin}${sharingStatus.share_url}`}
                readOnly
                className="font-mono text-xs"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={copyShareUrl}
                className="shrink-0"
              >
                <Copy className="h-4 w-4" />
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={openPublicView}
                className="shrink-0"
              >
                <ExternalLink className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Share Token Section (Optional) */}
        {sharingStatus?.is_public && sharingStatus.share_token && (
          <div className="space-y-3">
            <Label className="text-sm font-medium">
              {t("CONVERSATION_SHARING.SHARE_TOKEN")}
            </Label>
            <div className="flex gap-2">
              <Input
                value={sharingStatus.share_token}
                readOnly
                className="font-mono text-xs"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={copyShareToken}
                className="shrink-0"
              >
                <Copy className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-gray-600">
              {t("CONVERSATION_SHARING.TOKEN_DESCRIPTION")}
            </p>
          </div>
        )}

        {/* Info Section */}
        <Alert>
          <AlertDescription className="text-sm">
            {sharingStatus?.is_public
              ? t("CONVERSATION_SHARING.PUBLIC_INFO")
              : t("CONVERSATION_SHARING.PRIVATE_INFO")
            }
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
